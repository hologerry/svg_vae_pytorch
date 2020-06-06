import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

from dataloader import get_loader
# from models.image_vae import ImageVAE
from models.vae import ConditionalVAE
from models.svg_decoder import SVGLSTMDecoder, init_weights
# from models.svg_decoder import SVGMDNTop
from models import util_funcs
from options import (get_parser_basic, get_parser_image_vae,
                     get_parser_svg_decoder)
from data_utils.svg_utils import render

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_image_vae(opts):
    exp_dir = os.path.join("experiments", opts.experiment_name)
    sample_dir = os.path.join(exp_dir, "samples")
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    # res_dir = os.path.join(exp_dir, "results")
    log_dir = os.path.join(exp_dir, "logs")

    logfile = open(os.path.join(log_dir, "train_loss_log.txt"), 'w')
    val_logfile = open(os.path.join(log_dir, "val_loss_log.txt"), 'w')

    train_loader = get_loader(opts.data_root, opts.max_seq_len, opts.seq_feature_dim, opts.batch_size, opts.mode)
    val_loader = get_loader(opts.data_root, opts.max_seq_len, opts.seq_feature_dim, opts.batch_size, 'test')
    opts.kl_beta = 1.0 / len(train_loader)  # This is from PyTorch-VAE
    # model = ImageVAE(input_channels=opts.in_channel, output_channels=opts.out_channel,
    #                  num_categories=opts.num_categories, base_depth=opts.base_depth,
    #                  bottleneck_bits=opts.bottleneck_bits, free_bits=opts.free_bits,
    #                  kl_beta=opts.kl_beta, mode=opts.mode)
    model = ConditionalVAE(in_channels=opts.in_channel, num_classes=opts.num_categories, latent_dim=opts.bottleneck_bits, kl_beta=opts.kl_beta)

    if torch.cuda.is_available() and opts.multi_gpu:
        model = nn.DataParallel(model)
    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=opts.lr, betas=(opts.beta1, opts.beta2), eps=opts.eps, weight_decay=opts.weight_decay)

    if opts.tboard:
        writer = SummaryWriter(log_dir)

    for epoch in range(opts.init_epoch, opts.n_epochs):
        for idx, data in enumerate(train_loader):
            input_image = data['rendered'].to(device)
            # target_image = input_image.clone().detach()
            target_clss = data['class'].to(device)
            target_clss = F.one_hot(target_clss, num_classes=opts.num_categories).squeeze(dim=1)
            # input_image, target_clss = data
            # input_image = input_image.to(device)
            # target_clss = target_clss.to(device)
            output = model(input_image, target_clss)

            # ImageVAE
            # output_image = output['dec_out']
            # b_loss = output['b_loss'].mean()
            # rec_loss = output['rec_loss'].mean()
            # training_loss = output['training_loss'].mean()
            # img_rec_loss = output['img_rec_loss'].mean()

            # NOTE: b_loss, rec_loss and training_loss are negative huge, fall to nan
            # loss = b_loss + rec_loss + training_loss
            # loss = b_loss + rec_loss

            # ConditionalVAE
            output_image = output[0]
            if torch.cuda.is_available() and opts.multi_gpu:
                losses = model.module.loss_function(*output)
            else:
                losses = model.loss_function(*output)
            loss = losses['loss']
            rec_loss = losses['Reconstruction_Loss']
            b_loss = losses['KLD']

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batches_done = epoch * len(train_loader) + idx + 1

            message = (
                f"Epoch: {epoch}/{opts.n_epochs}, Batch: {idx}/{len(train_loader)}, "
                f"Loss: {loss.item():.6f}, "
                f"rec_loss: {rec_loss.item():.6f}, "
                f"b_loss: {b_loss.item():.6f}, "
                # f"training_loss: {training_loss.item():.6f}, "
                # f"img_rec_loss: {img_rec_loss.item():.6f}"
            )
            logfile.write(message + '\n')
            if batches_done % 50 == 0:
                print(message)

            if opts.tboard:
                writer.add_scalar('Loss/loss', loss.item(), batches_done)
                writer.add_scalar('Loss/rec_loss', rec_loss.item(), batches_done)
                writer.add_scalar('Loss/b_loss', b_loss.item(), batches_done)
                # writer.add_scalar('Loss/training_loss', training_loss.item(), batches_done)

            if opts.sample_freq > 0 and batches_done % opts.sample_freq == 0:
                img_sample = torch.cat((input_image.data, output_image.data), -2)
                save_file = os.path.join(sample_dir, f"train_epoch_{epoch}_batch_{batches_done}.png")
                save_image(img_sample, save_file, nrow=8, normalize=True)

            if opts.val_freq > 0 and batches_done % opts.val_freq == 0:
                val_loss = 0.0
                val_img_rec_loss = 0.0
                val_b_loss = 0.0
                with torch.no_grad():
                    for val_idx, val_data in enumerate(val_loader):
                        val_input_image = val_data['rendered'].to(device)
                        # val_target_image = val_input_image.clone().detach()
                        val_target_clss = val_data['class'].to(device)
                        val_target_clss = F.one_hot(val_target_clss, num_classes=opts.num_categories).squeeze(dim=1)
                        # val_input_image, val_target_clss = val_data
                        # val_input_image = val_input_image.to(device)
                        val_target_clss = val_target_clss.to(device)

                        val_output = model(val_input_image, val_target_clss)

                        val_output_image = val_output[0]
                        if torch.cuda.is_available() and opts.multi_gpu:
                            val_losses = model.module.loss_function(*val_output)
                        else:
                            val_losses = model.loss_function(*val_output)

                        val_loss += val_losses['loss'].item()
                        val_img_rec_loss += val_losses['Reconstruction_Loss'].mean()
                        val_b_loss += val_losses['KLD'].mean()

                        val_img_sample = torch.cat((val_input_image.data, val_output_image.data), -2)
                        val_save_file = os.path.join(sample_dir, f"val_epoch_{epoch}_batch_{batches_done}.png")
                        save_image(val_img_sample, val_save_file, nrow=8, normalize=True)

                    val_loss /= len(val_loader)
                    val_img_rec_loss /= len(val_loader)
                    val_b_loss /= len(val_loader)

                    if opts.tboard:
                        writer.add_scalar('VAL/loss', val_loss, batches_done)
                        writer.add_scalar('VAL/rec_loss', val_img_rec_loss, batches_done)
                        writer.add_scalar('VAL/b_loss', val_b_loss, batches_done)

                    val_msg = (
                        f"Epoch: {epoch}/{opts.n_epochs}, Batch: {idx}/{len(train_loader)}, "
                        f"Val loss: {val_loss: .6f}, "
                        f"Val image rec loss: {val_img_rec_loss: .6f}, "
                        f"Val kl loss: {val_b_loss: .6f}"
                    )

                    val_logfile.write(val_msg + "\n")
                    print(val_msg)

        if epoch % opts.ckpt_freq == 0:
            model_file = os.path.join(ckpt_dir, f"{opts.model_name}_{epoch}.pth")
            if torch.cuda.is_available() and opts.multi_gpu:
                torch.save(model.module.state_dict(), model_file)
            else:
                torch.save(model.state_dict(), model_file)

    logfile.close()
    val_logfile.close()


def train_svg_decoder(opts):
    # pass
    exp_dir = os.path.join("experiments", opts.experiment_name)
    sample_dir = os.path.join(exp_dir, "samples")
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    # res_dir = os.path.join(exp_dir, "results")
    log_dir = os.path.join(exp_dir, "logs")

    logfile = open(os.path.join(log_dir, "train_loss_log.txt"), 'w')
    val_logfile = open(os.path.join(log_dir, "val_loss_log.txt"), 'w')

    train_loader = get_loader(opts.data_root, opts.max_seq_len, opts.seq_feature_dim, opts.batch_size, opts.mode)
    val_loader = get_loader(opts.data_root, opts.max_seq_len, opts.seq_feature_dim, opts.batch_size, 'test')

    # image_vae = ImageVAE(input_channels=opts.in_channel, output_channels=opts.out_channel,
    #                      num_categories=opts.num_categories, base_depth=opts.base_depth,
    #                      bottleneck_bits=opts.bottleneck_bits, free_bits=opts.free_bits,
    #                      kl_beta=opts.kl_beta, mode=opts.mode)
    image_vae = ConditionalVAE(in_channels=opts.in_channel, num_classes=opts.num_categories, latent_dim=opts.bottleneck_bits, kl_beta=opts.kl_beta)

    svg_decoder = SVGLSTMDecoder(input_channels=opts.in_channel, output_channels=opts.out_channel,
                                 num_categories=opts.num_categories,
                                 bottleneck_bits=opts.bottleneck_bits, free_bits=opts.free_bits,
                                 kl_beta=opts.kl_beta, mode=opts.mode, max_sequence_length=opts.max_seq_len,
                                 hidden_size=opts.hidden_size, use_cls=opts.use_cls, dropout_p=opts.rec_dropout,
                                 twice_decoder=opts.twice_decoder, num_hidden_layers=opts.num_hidden_layers,
                                 feature_dim=opts.seq_feature_dim, ff_dropout=opts.ff_dropout)
    svg_decoder.apply(init_weights)
    # mdn_top_layer = SVGMDNTop(num_mixture=opts.num_mixture, seq_len=opts.max_seq_len, hidden_size=opts.hidden_size,
    #                           mode=opts.mode, mix_temperature=opts.mix_temperature,
    #                           gauss_temperature=opts.gauss_temperature, dont_reduce=opts.dont_reduce_loss)

    if torch.cuda.is_available() and opts.multi_gpu:
        image_vae = nn.DataParallel(image_vae)
        svg_decoder = nn.DataParallel(svg_decoder)
        # mdn_top_layer = nn.DataParallel(mdn_top_layer)
    image_vae = image_vae.to(device)
    svg_decoder = svg_decoder.to(device)
    # mdn_top_layer = mdn_top_layer.to(device)

    image_vae.load_state_dict(torch.load(opts.vae_ckpt_path, map_location=device))

    # TODO: joint train
    image_vae.eval()

    optimizer = Adam(svg_decoder.parameters(), lr=opts.lr, betas=(opts.beta1, opts.beta2), eps=opts.eps, weight_decay=opts.weight_decay)

    if opts.tboard:
        writer = SummaryWriter(log_dir)

    tearcher_force_ratio = 1.0

    for epoch in range(opts.init_epoch, opts.n_epochs):
        for idx, data in enumerate(train_loader):
            input_image = data['rendered'].to(device)
            # target_image = input_image.clone().detach()
            target_clss = data['class'].to(device)
            target_clss = F.one_hot(target_clss, num_classes=opts.num_categories).squeeze(dim=1)
            target_seq = data['sequence'].to(device)
            gt_target_seq = target_seq.clone().detach()
            # sequence first batch second
            target_seq = target_seq.reshape(target_seq.size(1), target_seq.size(0), target_seq.size(2))
            target_seq = util_funcs.shift_right(target_seq)
            # outputs = torch.zeros(target_seq.size(0), target_seq.size(1), opts.hidden_size).to(device)
            outputs = torch.zeros(target_seq.size(0), target_seq.size(1), target_seq.size(2)).to(device)

            vae_output = image_vae(input_image, target_clss)
            sampled_bottleneck = vae_output[2]  # z
            vae_output_image = vae_output[0]
            inpt = target_seq[0]

            init_state = svg_decoder.init_state_input(sampled_bottleneck)
            hidden, cell = init_state['hidden'], init_state['cell']

            trg_len = target_seq.size(0)

            for t in range(1, trg_len):
                decoder_output = svg_decoder(inpt, sampled_bottleneck, target_clss, hidden, cell)
                # output, hidden, cell = decoder_output['output'], decoder_output['hidden'], decoder_output['cell']
                output, hidden, cell = decoder_output['predict'], decoder_output['hidden'], decoder_output['cell']
                outputs[t] = output

                # print(output.size())
                teacher_force = random.random() < tearcher_force_ratio

                inpt = target_seq[t] if (teacher_force and opts.mode == 'train') else output.clone().detach()
                # print(inpt.size())

            # top_output = mdn_top_layer(outputs)

            # svg_losses = mdn_top_layer.svg_loss(top_output, target_seq)
            # mdn_loss, softmax_xent_loss = svg_losses['mdn_loss'], svg_losses['softmax_xent_loss']
            # loss = mdn_loss + softmax_xent_loss
            outputs = outputs[1:]
            target_seq = target_seq[1:]
            svg_losses = svg_decoder.decoder_loss(outputs, target_seq)
            loss = svg_losses['loss']
            mse_loss, softmax_xent_loss = svg_losses['mse_loss'], svg_losses['xent_loss']

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            batches_done = epoch * len(train_loader) + idx + 1

            message = (
                f"Epoch: {epoch}/{opts.n_epochs}, Batch: {idx}/{len(train_loader)}, "
                f"Loss: {loss.item():.6f}, "
                f"mse_loss: {mse_loss.item():.6f}, "
                f"softmax_xent_loss: {softmax_xent_loss.item():.6f}"
            )
            logfile.write(message + '\n')
            if batches_done % 40 == 0:
                print(message)

            if opts.tboard:
                writer.add_scalar('Loss/loss', loss.item(), batches_done)
                writer.add_scalar('Loss/mse_loss', mse_loss.item(), batches_done)
                writer.add_scalar('Loss/softmax_xent_loss', softmax_xent_loss.item(), batches_done)

            if opts.sample_freq > 0 and batches_done % opts.sample_freq == 0:
                svg_dec_out = outputs.clone().detach()
                svg_dec_out = svg_dec_out.view(svg_dec_out.size(1), svg_dec_out.size(0), svg_dec_out.size(2))  # batch first
                for i, one_seq in enumerate(svg_dec_out):
                    svg = render(one_seq.cpu().numpy())
                    cur_svg_file = os.path.join(sample_dir, f"train_epoch_{epoch}_batch_{batches_done}_output_svg_{i}.svg")
                    with open(cur_svg_file, 'w') as f:
                        f.write(svg)

                svg_target = gt_target_seq.clone().detach()
                for i, one_gt_seq in enumerate(svg_target):
                    gt_svg = render(one_gt_seq.cpu().numpy())
                    cur_svg_file = os.path.join(sample_dir, f"train_epoch_{epoch}_batch_{batches_done}_gt_svg_{i}.svg")
                    with open(cur_svg_file, 'w') as f:
                        f.write(gt_svg)
                img_sample = torch.cat((input_image.data, vae_output_image.data), -2)
                save_file = os.path.join(sample_dir, f"train_epoch_{epoch}_batch_{batches_done}_input_vae.png")
                save_image(img_sample, save_file, nrow=8, normalize=True)

            if opts.val_freq > 0 and batches_done % opts.val_freq == 0:
                val_loss_value = 0.0
                with torch.no_grad():
                    for val_idx, val_data in enumerate(val_loader):
                        if val_idx >= 20:
                            break
                        val_input_image = val_data['rendered'].to(device)
                        # val_target_image = val_input_image.clone().detach()
                        val_target_clss = val_data['class'].to(device)
                        val_target_clss = F.one_hot(val_target_clss, num_classes=opts.num_categories).squeeze(dim=1)
                        val_gt_target_seq = val_data['sequence'].to(device)
                        # sequence first batch second
                        val_target_seq = val_gt_target_seq.clone().detach()
                        val_target_seq = val_target_seq.reshape(val_target_seq.size(1), val_target_seq.size(0), val_target_seq.size(2))
                        val_target_seq = util_funcs.shift_right(val_target_seq)

                        val_outputs = torch.zeros(val_target_seq.size(0), val_target_seq.size(1), val_target_seq.size(2)).to(device)

                        vae_output = image_vae(val_input_image, val_target_clss)
                        val_sampled_bottleneck = vae_output[2]
                        val_inpt = val_target_seq[0]

                        val_init_state = svg_decoder.init_state_input(val_sampled_bottleneck)
                        val_hidden, val_cell = val_init_state['hidden'], val_init_state['cell']
                        val_trg_len = val_target_seq.size(0)
                        for val_t in range(1, val_trg_len):
                            val_decoder_output = svg_decoder(val_inpt, val_sampled_bottleneck, val_target_clss, val_hidden, val_cell)
                            val_output, val_hidden, val_cell = val_decoder_output['predict'], val_decoder_output['hidden'], val_decoder_output['cell']
                            val_outputs[val_t] = val_output

                            val_inpt = val_output.clone().detach()

                        # val_top_output = mdn_top_layer(val_outputs, 'test')
                        val_outputs = val_outputs[1:]
                        val_target_seq = val_target_seq[1:]
                        val_svg_losses = svg_decoder.decoder_loss(val_outputs, val_target_seq)
                        val_loss = val_svg_losses['loss']
                        val_loss_value += val_loss.item()
                        # val_mse_loss, val_softmax_xent_loss = val_svg_losses['mse_loss'], val_svg_losses['xent_loss']

                        val_svg_dec_out = val_outputs.clone().detach()
                        val_svg_dec_out = val_svg_dec_out.view(val_svg_dec_out.size(1), val_svg_dec_out.size(0), val_svg_dec_out.size(2))  # batch first
                        for i, val_one_seq in enumerate(val_svg_dec_out):
                            val_svg = render(val_one_seq.cpu().numpy())
                            val_cur_svg_file = os.path.join(sample_dir, f"val_epoch_{epoch}_batch_{batches_done}_output_svg_{i}.svg")
                            with open(val_cur_svg_file, 'w') as f:
                                f.write(val_svg)

                        val_svg_target = val_gt_target_seq.clone().detach()
                        for i, one_gt_seq in enumerate(val_svg_target):
                            gt_svg = render(one_gt_seq.cpu().numpy())
                            cur_svg_file = os.path.join(sample_dir, f"val_epoch_{epoch}_batch_{batches_done}_gt_svg_{i}.svg")
                            with open(cur_svg_file, 'w') as f:
                                f.write(gt_svg)

                        val_img_sample = torch.cat((input_image.data, vae_output_image.data), -2)
                        val_save_file = os.path.join(sample_dir, f"val_epoch_{epoch}_batch_{batches_done}_input_vae.png")
                        save_image(val_img_sample, val_save_file, nrow=8, normalize=True)

                    val_loss_value /= 20
                    val_msg = (
                        f"Epoch: {epoch}/{opts.n_epochs}, Batch: {idx}/{len(train_loader)}, "
                        f"MSE+Soft loss: {val_loss_value: .6f}"
                    )
                    val_logfile.write(val_msg + "\n")
                    print(val_msg)

        if epoch % opts.ckpt_freq == 0:
            decoder_model_file = os.path.join(ckpt_dir, f"{opts.model_name}_lstm_{epoch}.pth")
            # top_model_file = os.path.join(ckpt_dir, f"{opts.model_name}_top_{epoch}.pth")
            if torch.cuda.is_available() and opts.multi_gpu:
                torch.save(svg_decoder.module.state_dict(), decoder_model_file)
                # torch.save(mdn_top_layer.module.state_dict(), top_model_file)
            else:
                torch.save(svg_decoder.state_dict(), decoder_model_file)
                # torch.save(mdn_top_layer.state_dict(), top_model_file)

    logfile.close()
    val_logfile.close()


def train(opts):
    if opts.model_name == 'image_vae':
        train_image_vae(opts)
    elif opts.model_name == 'svg_decoder':
        train_svg_decoder(opts)
    else:
        raise NotImplementedError


def test_image_vae(opts):
    pass


def test_svg_decoder(opts):
    pass


def test(opts):
    if opts.model_name == 'image_vae':
        test_image_vae(opts)
    elif opts.model_name == 'svg_decoder':
        test_svg_decoder(opts)
    else:
        raise NotImplementedError


def main():
    basic_opts = get_parser_basic().parse_args()
    if basic_opts.model_name == 'image_vae':
        opts = get_parser_image_vae().parse_args()
    elif basic_opts.model_name == 'svg_decoder':
        opts = get_parser_svg_decoder().parse_args()
    else:
        raise NotImplementedError

    opts.experiment_name = opts.experiment_name + '_' + opts.model_name

    os.makedirs("experiments", exist_ok=True)

    debug = True

    if opts.mode == 'train':
        # Create directories
        experiment_dir = os.path.join("experiments", opts.experiment_name)
        os.makedirs(experiment_dir, exist_ok=debug)  # False to prevent multiple train run by mistake
        os.makedirs(os.path.join(experiment_dir, "samples"), exist_ok=True)
        os.makedirs(os.path.join(experiment_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(experiment_dir, "results"), exist_ok=True)
        os.makedirs(os.path.join(experiment_dir, "logs"), exist_ok=True)

        print(f"Training on experiment {opts.experiment_name}...")
        # Dump options
        with open(os.path.join(experiment_dir, "opts.txt"), "w") as f:
            for key, value in vars(opts).items():
                f.write(str(key) + ": " + str(value) + "\n")
        train(opts)
    elif opts.mode == 'test':
        print(f"Testing on experiment {opts.experiment_name}...")
        test(opts)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
