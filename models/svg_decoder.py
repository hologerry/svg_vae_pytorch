import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

import models.util_funcs as util_funcs
from models.image_vae import ImageVAE


class SVGLSTMDecoder(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, num_categories=52,
                 base_depth=32, bottleneck_bits=32, free_bits=0.15,
                 kl_beta=300, mode='train', sg_bottleneck=True, max_sequence_length=51,
                 hidden_size=1024, use_cls=True, dropout_p=0.5,
                 twice_decoder=False, num_hidden_layers=4, feature_dim=10):
        super().__init__()
        self.bottleneck_bits = bottleneck_bits
        # self.num_categories = num_categories
        self.sg_bottleneck = sg_bottleneck
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        if twice_decoder:
            self.hidden_size = self.hidden_size * 2
        self.unbottleneck_dim = self.hidden_size * 2

        self.unbotltenecks = [nn.Linear(bottleneck_bits, self.unbottleneck_dim) for _ in range(self.num_hidden_layers)]

        self.input_dim = feature_dim + bottleneck_bits + num_categories
        self.pre_lstm_fc = nn.Linear(self.input_dim, self.hidden_size)
        self.pre_lstm_ac = nn.Tanh()

        self.dropout = nn.Dropout(dropout_p)
        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, self.num_hidden_layers, dropout=dropout_p)

    def init_state_input(self, sampled_bottleneck):
        init_state_hidden = []
        init_state_cell = []
        for i in range(self.num_hidden_layers):
            unbottleneck = self.unbotltenecks[i](sampled_bottleneck)
            (h0, c0) = unbottleneck[:, :self.unbottleneck_dim//2], unbottleneck[:, self.unbottleneck_dim//2:]
            init_state_hidden.append(h0.unsqueeze(0))
            init_state_cell.append(c0.unsqueeze(0))
        init_state_hidden = torch.cat(init_state_hidden, dim=0)
        init_state_cell = torch.cat(init_state_cell, dim=0)
        return (init_state_hidden, init_state_cell)

    def forward(self, inpt, sampled_bottleneck, clss, hidden, cell):
        clss = F.one_hot(clss, num_categories).float().squeeze(1)
        inpt = torch.cat([inpt, sampled_bottleneck, clss], dim=-1)
        inpt = self.pre_lstm_ac(self.pre_lstm_fc(inpt)).unsqueeze(0)
        inpt = self.dropout(inpt)
        output, (hidden, cell) = self.rnn(inpt, (hidden, cell))
        return output, (hidden, cell)


def lognormal(y, mean, logstd, logsqrttwopi):
    return -0.5 * (y - mean) / torch.exp(logstd) ** 2 - logstd - logsqrttwopi


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

    def forward(self, decoder_output):
        ret = self.fc(decoder_output)
        if self.hard or self.mode != 'train':
            # apply temperature, do softmax
            command = self.iden(ret[..., :self.command_len]) / self.mix_temperature
            command = torch.exp(command - torch.max(command, dim=-1, keepdim=True))
            command = command / torch.sum(command, dim=-1, keepdim=True)

            # sample from the given probs, this is the same as get_pi_idx
            # and already returns not soft prob
            # [seq_len, batch, 4]
            command = Categorical(probs=command).sample()
            # [seq_len, batch]
            command = F.one_hot(command, self.command_len)

            arguments = ret[..., self.command_len:]
            # args are [seq_len, batch, 6*3*num_mix], and get [seq_len*batch*6, 3*num_mix]
            arguments = arguments.reshape([-1, 3 * self.num_mix])
            out_logmix, out_mean, out_logstd = self.get_mdn_coef(arguments)
            # these are [seq_len*batch*6, num_mix]

            # apply temp to logmix
            out_logmix = self.identity(out_logmix) / self.mix_temperature
            out_logmix = torch.exp(out_logmix - torch.max(out_logmix, dim=-1, keepdim=True))
            out_logmix = out_logmix / torch.sum(out_logmix, dim=-1, keepdim=True)
            # get_pi_idx
            out_logmix = Categorical(probs=out_logmix).sample()
            # [seq_len*batch*6]
            out_logmix = out_logmix.long()
            out_logmix = torch.cat([torch.arange(out_logmix.size(0)), out_logmix], dim=-1)

            chosen_mean = torch.gather(out_mean, 0, out_logmix)
            chosen_logstd = torch.gather(out_logstd, 0, out_logmix)

            rand_gaussian = (torch.randn(chosen_mean.size()) * torch.sqrt(self.gauss_temperature))
            arguments = chosen_mean + torch.exp(chosen_logstd) * rand_gaussian

            batch_size = command.size(1)
            arguments = arguments.view(-1, batch_size, 6)

            # concat with the command we picked
            ret = torch.cat([command, arguments], dim=-1)

        return ret

    def get_mdn_coef(self, arguments):
        """Compute mdn coefficient, aka, split arguments to 3 chunck with size num_mix"""
        logmix, mean, logstd = torch.split(arguments, self.num_mix, dim=-1)
        logmix = logmix - torch.logsumexp(logmix, -1, keepdim=True)
        return logmix, mean, logstd

    def get_mdn_loss(self, logmix, mean, logstd, args_flat, batch_mask):
        """Compute MDN loss term for svg decoder model."""
        logsqrttwopi = math.log(math.sqrt(2.0 * math.pi))

        lognorm = util_funcs.lognormal(args_flat, mean, logstd, logsqrttwopi)
        v = logmix + lognorm
        v = torch.logsumexp(v, 1, keepdim=True)
        v = v.reshape([self.seq_len, -1, self.arg_len])
        v = v * batch_mask
        if self.dont_reduce:
            return -torch.mean(torch.sum(v, dim=2), dim=[0, 1], keepdim=True)
        return -torch.mean(torch.sum(v, dim=2))

    def svg_loss(self, mdn_top_out, target):
        """Compute """
        # target already in 10-dim mode, no need to mdn
        target_commands = target[..., :self.command_len]
        target_args = target[..., self.command_len:]

        predict_commands = mdn_top_out[..., :self.command_len]
        predict_args = mdn_top_out[..., self.command_len:]
        # [seq_len, batch, 6*3*num_mix]
        predict_args = predict_args.reshape([-1, 3 * self.num_mix])
        out_logmix, out_mean, out_logstd = self.get_mdn_coef(predict_args)

        # create a mask for elements to ignore on it
        masktemplate = torch.Tensor([[0., 0., 0., 0., 0., 0.],
                                     [0., 0., 0., 0., 1., 1.],
                                     [0., 0., 0., 0., 1., 1.],
                                     [1., 1., 1., 1., 1., 1.]])
        mask = torch.matmul(target_commands, masktemplate)
        target_args_flat = target_args.reshape([-1, 1])
        mdn_loss = self.get_mdn_loss(out_logmix, out_mean, out_logstd, target_args_flat, mask)

        softmax_xent_loss = F.softmax(F.binary_cross_entropy_with_logits(predict_commands, target_commands), dim=-1)

        if self.dont_reduce:
            softmax_xent_loss = torch.mean(softmax_xent_loss, dim=[1, 2], keepdim=True)
        else:
            softmax_xent_loss = torch.mean(softmax_xent_loss)

        return mdn_loss, softmax_xent_loss


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 2
    seq_len = 51
    feature_dim = 10  # like vocab_size
    bottleneck_bits = 32
    num_categories = 52
    hidden_size = 1024
    sg_bottleneck = True
    tearcher_force_ratio = 1.0

    image = torch.randn((batch_size, 1, 64, 64)).to(device)
    clss = torch.randint(low=0, high=num_categories, size=(batch_size, 1), dtype=torch.long).to(device)

    image_vae = ImageVAE().to(device)
    svg_decoder = SVGLSTMDecoder().to(device)
    if sg_bottleneck:
        image_vae = image_vae.eval().to(device)

    sampled_bottleneck, _, _ = image_vae(image, clss)

    trg = torch.randn((seq_len, batch_size, feature_dim)).to(device)  # [seq_len, batch_size, feature_dim]
    trg_len = trg.size(0)

    outputs = torch.zeros(trg_len, batch_size, hidden_size).to(device)

    inpt = trg[0, :, :]
    print(inpt.size())

    hidden, cell = svg_decoder.init_state_input(sampled_bottleneck)

    for t in range(1, trg_len):
        output, (hidden, cell) = svg_decoder(inpt, sampled_bottleneck, clss, hidden, cell)
        outputs[t] = output
        teacher_force = random.random() < tearcher_force_ratio
        # print(output.size())
        _, topi = output.topk(feature_dim)
        inpt = trg[t] if teacher_force else topi.squeeze(0).detach().float()
        # print(inpt.size())

    print(outputs.size())

    mdn_top_layer = SVGMDNTop()

    top_output = mdn_top_layer(outputs)

    mdn_loss, softmax_xent_loss = mdn_top_layer.svg_loss(top_output, trg)
    print(mdn_loss, softmax_xent_loss)
