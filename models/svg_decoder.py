import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

import models.util_funcs as util_funcs
from models.vae import ConditionalVAE


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


class SVGLSTMDecoder(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, num_categories=52,
                 bottleneck_bits=32, free_bits=0.15,
                 kl_beta=300, mode='train', max_sequence_length=51,
                 hidden_size=1024, use_cls=True, dropout_p=0.5,
                 twice_decoder=False, num_hidden_layers=4, feature_dim=10, ff_dropout=True):
        super().__init__()
        self.mode = mode
        self.bottleneck_bits = bottleneck_bits
        self.num_categories = num_categories
        # self.sg_bottleneck = sg_bottleneck
        self.command_len = 4
        self.arg_len = 6
        assert self.command_len + self.arg_len == feature_dim

        self.ff_dropout = ff_dropout
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        if twice_decoder:
            self.hidden_size = self.hidden_size * 2
        self.unbottleneck_dim = self.hidden_size * 2

        self.unbotltenecks = nn.ModuleList([nn.Linear(bottleneck_bits, self.unbottleneck_dim) for _ in range(self.num_hidden_layers)])

        self.input_dim = feature_dim + bottleneck_bits + num_categories
        self.pre_lstm_fc = nn.Linear(self.input_dim, self.hidden_size)
        self.pre_lstm_ac = nn.Tanh()
        if self.ff_dropout:
            self.dropout = nn.Dropout(dropout_p)
        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, self.num_hidden_layers, dropout=dropout_p)
        self.predict_fc = nn.Linear(self.hidden_size, feature_dim)

    def init_state_input(self, sampled_bottleneck):
        init_state_hidden = []
        init_state_cell = []
        for i in range(self.num_hidden_layers):
            unbottleneck = self.unbotltenecks[i](sampled_bottleneck)
            (h0, c0) = unbottleneck[:, :self.unbottleneck_dim // 2], unbottleneck[:, self.unbottleneck_dim // 2:]
            init_state_hidden.append(h0.unsqueeze(0))
            init_state_cell.append(c0.unsqueeze(0))
        init_state_hidden = torch.cat(init_state_hidden, dim=0)
        init_state_cell = torch.cat(init_state_cell, dim=0)
        init_state = {}
        init_state['hidden'] = init_state_hidden
        init_state['cell'] = init_state_cell
        return init_state

    def decoder_loss(self, decoder_predict, target, mode='train'):
        target_commands = target[..., :self.command_len]
        target_args = target[..., self.command_len:]
        predict_commands = decoder_predict[..., :self.command_len]
        predict_args = decoder_predict[..., self.command_len:]
        softmax_xent_loss = torch.sum(- target_commands * F.log_softmax(predict_commands, -1), -1)
        softmax_xent_loss = torch.mean(softmax_xent_loss)
        mse_loss = F.mse_loss(predict_args, target_args)
        loss = {}
        loss['loss'] = softmax_xent_loss * 1.0 + mse_loss * 2.0
        loss['xent_loss'] = softmax_xent_loss
        loss['mse_loss'] = mse_loss
        return loss

    def forward(self, inpt, sampled_bottleneck, clss, hidden, cell):
        clss = clss.float()
        if inpt.size(-1) != self.hidden_size:  # train and first time step in test
            # clss = F.one_hot(clss, self.num_categories).to(device=inpt.device).float().squeeze(1)
            inpt = torch.cat([inpt, sampled_bottleneck, clss], dim=-1)  # [batch_size, 10 + 32 + 52]
            inpt = self.pre_lstm_ac(self.pre_lstm_fc(inpt))
            inpt = inpt.unsqueeze(dim=0)
            # print(inpt.size())
        if self.ff_dropout:
            inpt = self.dropout(inpt)
        output, (hidden, cell) = self.rnn(inpt, (hidden, cell))
        predict = self.predict_fc(output.squeeze(0))
        decoder_output = {}
        decoder_output['predict'] = predict
        decoder_output['output'] = output
        decoder_output['hidden'] = hidden
        decoder_output['cell'] = cell
        return decoder_output


class SVGMDNTop(nn.Module):
    """
    Apply the Mixture Nensity Network on the top of the LSTM ouput
    Input:
        body_output: outputs from LSTM [seq_len, batch, hidden_size]
    Output:
        The MDN output. predict mode (hard=True): [seq_len, batch, 10] feature_dim = 10
        train mode or head=False: [seq_len, batch, 4 + 6 * self.num_mix * 3]
    """
    def __init__(self, num_mixture=50, seq_len=51, hidden_size=1024, hard=False, mode='train',
                 mix_temperature=0.0001, gauss_temperature=0.0001, dont_reduce=False):
        super().__init__()
        self.num_mix = num_mixture
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.command_len = 4
        self.arg_len = 6
        self.output_channel = self.command_len + self.arg_len * self.num_mix * 3

        self.hard = hard
        self.mode = mode
        self.mix_temperature = mix_temperature
        self.gauss_temperature = gauss_temperature

        self.dont_reduce = dont_reduce

        self.fc = nn.Linear(self.hidden_size, self.output_channel)
        self.identity = nn.Identity()

    def forward(self, decoder_output, mode='train'):
        ret = self.fc(decoder_output)
        if self.hard or mode != 'train':
            # apply temperature, do softmax
            command = self.identity(ret[..., :self.command_len]) / self.mix_temperature
            command_max = torch.max(command, dim=-1, keepdim=True)[0]
            command = torch.exp(command - command_max)
            command = command / torch.sum(command, dim=-1, keepdim=True)

            # sample from the given probs, this is the same as get_pi_idx
            # and already returns not soft prob
            # [seq_len, batch, command_len]
            command = Categorical(probs=command).sample()
            # [seq_len, batch]
            command = F.one_hot(command, self.command_len).to(decoder_output.device).float()
            # print(command.size())

            arguments = ret[..., self.command_len:]
            # args are [seq_len, batch, 6*3*num_mix], and get [seq_len*batch*6, 3*num_mix]
            arguments = arguments.reshape([-1, 3 * self.num_mix])
            mdn_coef = self.get_mdn_coef(arguments)
            out_logmix, out_mean, out_logstd = mdn_coef['logmix'], mdn_coef['mean'], mdn_coef['logstd']
            # these are [seq_len*batch*6, num_mix]

            # apply temp to logmix
            out_logmix = self.identity(out_logmix) / self.mix_temperature
            out_logmix_max = torch.max(out_logmix, dim=-1, keepdim=True)[0]
            out_logmix = torch.exp(out_logmix - out_logmix_max)
            out_logmix = out_logmix / torch.sum(out_logmix, dim=-1, keepdim=True)
            # get_pi_idx
            out_logmix = Categorical(probs=out_logmix).sample()
            # [seq_len*batch*arg_len]
            out_logmix = out_logmix.long().unsqueeze(1)
            out_logmix = torch.cat([torch.arange(out_logmix.size(0), device=decoder_output.device).unsqueeze(1), out_logmix], dim=-1)
            # [seq_len*batch*arg_len, 2]
            chosen_mean = [out_mean[i[0], i[1]] for i in out_logmix]
            chosen_logstd = [out_logstd[i[0], i[1]] for i in out_logmix]
            chosen_mean = torch.tensor(chosen_mean, device=decoder_output.device)
            chosen_logstd = torch.tensor(chosen_logstd, device=decoder_output.device)

            rand_gaussian = (torch.randn(chosen_mean.size(), device=decoder_output.device) * math.sqrt(self.gauss_temperature))
            arguments = chosen_mean + torch.exp(chosen_logstd) * rand_gaussian

            batch_size = command.size(1)
            arguments = arguments.reshape(-1, batch_size, self.arg_len)  # [seg_len, batch, arg_len]

            # concat with the command we picked
            ret = torch.cat([command, arguments], dim=-1)

        return ret

    def get_mdn_coef(self, arguments):
        """Compute mdn coefficient, aka, split arguments to 3 chunck with size num_mix"""
        logmix, mean, logstd = torch.split(arguments, self.num_mix, dim=-1)
        logmix = logmix - torch.logsumexp(logmix, -1, keepdim=True)
        mdn_coef = {}
        mdn_coef['logmix'] = logmix
        mdn_coef['mean'] = mean
        mdn_coef['logstd'] = logstd
        return mdn_coef

    def get_mdn_loss(self, logmix, mean, logstd, args_flat, batch_mask):
        """Compute MDN loss term for svg decoder model."""
        logsqrttwopi = math.log(math.sqrt(2.0 * math.pi))
        # print('logmix min', torch.min(logmix))
        # print('mean min', torch.min(mean))
        # print('logstd min', torch.min(logstd))
        lognorm = util_funcs.lognormal(args_flat, mean, logstd, logsqrttwopi)
        # print('lognorm min', torch.min(lognorm))
        v = logmix + lognorm
        # print('v min', torch.min(v))
        v = torch.logsumexp(v, 1, keepdim=True)
        # print('v logsumexp min', torch.min(v))
        v = v.reshape([self.seq_len, -1, self.arg_len])
        v = v * batch_mask
        if self.dont_reduce:
            return -torch.mean(torch.sum(v, dim=2), dim=[0, 1], keepdim=True)
        return -torch.mean(torch.sum(v, dim=2))

    def svg_loss(self, mdn_top_out, target, mode='train'):
        """Compute loss for svg decoder model"""
        assert mode == 'train', "Need compute loss in train mode"
        # target already in 10-dim mode, no need to mdn
        target_commands = target[..., :self.command_len]
        target_args = target[..., self.command_len:]

        # in train mode the mdn_top_out has size [seq_len, batch, mdn_output_channel]
        predict_commands = mdn_top_out[..., :self.command_len]
        predict_args = mdn_top_out[..., self.command_len:]
        # [seq_len, batch, 6*3*num_mix]
        predict_args = predict_args.reshape([-1, 3 * self.num_mix])
        mdn_coef = self.get_mdn_coef(predict_args)
        out_logmix, out_mean, out_logstd = mdn_coef['logmix'], mdn_coef['mean'], mdn_coef['logstd']

        # create a mask for elements to ignore on it
        masktemplate = torch.Tensor([[0., 0., 0., 0., 0., 0.],
                                     [0., 0., 0., 0., 1., 1.],
                                     [0., 0., 0., 0., 1., 1.],
                                     [1., 1., 1., 1., 1., 1.]]).to(target_commands.device)
        mask = torch.matmul(target_commands, masktemplate)
        target_args_flat = target_args.reshape([-1, 1])
        mdn_loss = self.get_mdn_loss(out_logmix, out_mean, out_logstd, target_args_flat, mask)
        # print('mdn loss min', torch.min(mdn_loss))
        softmax_xent_loss = torch.sum(- target_commands * F.log_softmax(predict_commands, -1), -1)

        if self.dont_reduce:
            softmax_xent_loss = torch.mean(softmax_xent_loss, dim=[1, 2], keepdim=True)
        else:
            softmax_xent_loss = torch.mean(softmax_xent_loss)

        svg_losses = {}
        svg_losses['mdn_loss'] = mdn_loss
        svg_losses['softmax_xent_loss'] = softmax_xent_loss
        return svg_losses


if __name__ == "__main__":
    from data_utils.svg_utils import render
    from dataloader import get_loader
    # from torchvision.utils import save_image
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 2
    seq_len = 51
    feature_dim = 10  # like vocab_size
    bottleneck_bits = 32
    num_categories = 52
    hidden_size = 1024
    sg_bottleneck = True
    tearcher_force_ratio = 1.0

    mode = 'train'

    image = torch.randn((batch_size, 1, 64, 64)).to(device)
    clss = torch.randint(low=0, high=num_categories, size=(batch_size, 1), dtype=torch.long).to(device)
    clss = F.one_hot(clss, num_classes=num_categories).squeeze(dim=1)
    image_vae = ConditionalVAE(1, num_categories, bottleneck_bits, mode=mode).to(device)
    svg_decoder = SVGLSTMDecoder(mode=mode).to(device)
    if sg_bottleneck:
        image_vae = image_vae.eval().to(device)

    vae_output = image_vae(image, clss)
    sampled_bottleneck = vae_output[2]

    trg = torch.randn((seq_len, batch_size, feature_dim)).to(device)  # [seq_len, batch_size, feature_dim]
    trg = util_funcs.shift_right(trg)
    trg_len = trg.size(0)

    outputs = torch.zeros(trg_len, batch_size, hidden_size).to(device)
    if mode == 'train':
        inpt = trg[0, :, :]
    else:
        inpt = torch.zeros(1, batch_size, hidden_size).to(device)
    # print(inpt.size())

    init_state = svg_decoder.init_state_input(sampled_bottleneck)
    hidden, cell = init_state['hidden'], init_state['cell']

    for t in range(1, trg_len):
        decoder_output = svg_decoder(inpt, sampled_bottleneck, clss, hidden, cell)
        output, hidden, cell = decoder_output['output'], decoder_output['hidden'], decoder_output['cell']
        outputs[t] = output

        # print(output.size())
        teacher_force = random.random() < tearcher_force_ratio

        inpt = trg[t] if (teacher_force and mode == 'train') else output.clone().detach()
        # print(inpt.size())

    # print("rnn output", outputs.size())

    mdn_top_layer = SVGMDNTop(mode=mode)
    mdn_top_layer = mdn_top_layer.to(device)

    top_output = mdn_top_layer(outputs, 'test')
    # print("top output", top_output.size())
    # if mode == 'train':
    #     svg_losses = mdn_top_layer.svg_loss(top_output, trg)
    #     mdn_loss, softmax_xent_loss = svg_losses['mdn_loss'], svg_losses['softmax_xent_loss']
    #     print(mdn_loss, softmax_xent_loss)

    svg_decoder_output = top_output.clone().detach()
    # print("decoder output", svg_decoder_output.size())
    svg_decoder_output = svg_decoder_output.reshape(svg_decoder_output.size(1), svg_decoder_output.size(0), svg_decoder_output.size(2))  # batch first

    print("output size", svg_decoder_output.size())
    output_html = render(svg_decoder_output[0].cpu().numpy())

    # print(output_html)

    root_path = 'svg_vae_data/glyph_pkl_dataset_10'
    max_seq_len = 51
    seq_feature_dim = 10
    batch_size = 2

    loader = get_loader(root_path, max_seq_len, seq_feature_dim, batch_size, 'test')

    for idx, batch in enumerate(loader):
        if idx > 0:
            break
        # print('class', batch['class'].size())
        # print('seq_len', batch['seq_len'].size())
        # print('sequence', batch['sequence'].size())
        # print('rendered', batch['rendered'].size())
        # print(torch.max(batch['rendered']))
        input_image = batch['rendered'].to(device)
        sequence = batch['sequence'].to(device)
        output_svg = render(sequence[0].cpu().numpy())
        print(output_svg)
        val_cur_svg_file = "gt_svg.svg"
        with open(val_cur_svg_file, 'w') as f:
            f.write(output_svg)

        # img_sample = input_image.data
        # save_file = "input.png"
        # save_image(img_sample, save_file, nrow=8, normalize=True)
