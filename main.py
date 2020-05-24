import os

import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.utils import save_image

from dataloader import get_loader
from models.image_vae import ImageVAE
from models.svg_decoder import SVGLSTMDecoder, SVGMDNTop
from options import (get_parser_basic, get_parser_image_vae,
                     get_parser_svg_decoder)

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
    val_loader = get_loader(opts.data_root, opts.max_seq_len, opts.seq_feature_dim, 1, 'test')

    model = ImageVAE(input_channels=opts.in_channel, output_channels=opts.out_channel,
                     num_categories=opts.num_categories, base_depth=opts.base_depth,
                     bottleneck_bits=opts.bottleneck_bits, free_bits=opts.free_bits,
                     kl_beta=opts.kl_beta, mode=opts.mode)

    if torch.cuda.is_available() and opts.multi_gpu:
        model = nn.DataParallel(model)
    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=opts.lr, betas=(opts.beta1, opts.beta2), eps=opts.eps, weight_decay=opts.weight_decay)

    for epoch in range(opts.init_epoch, opts.n_epochs):
        for idx, data in enumerate(train_loader):
            input_image = data['rendered'].to(device)
            target_image = input_image.detach().clone()
            target_clss = data['class'].to(device)
            output = model(input_image, target_clss)

            output_image = output['dec_out']
            b_loss = output['b_loss'].mean()
            rec_loss = output['rec_loss'].mean()
            training_loss = output['training_loss'].mean()
            img_rec_loss = output['img_rec_loss'].mean()

            # TODO: b_loss, rec_loss and training_loss are negative huge, fall to nan
            # loss = b_loss + rec_loss + training_loss + img_rec_loss
            loss = b_loss + img_rec_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batches_done = epoch * len(train_loader) + idx

            message = (
                f"Epoch: {epoch}/{opts.n_epochs}, Batch: {idx}/{len(train_loader)}, "
                f"Loss: {loss.item():.6f}, "
                f"b_loss: {b_loss.item():.6f}, "
                f"rec_loss: {rec_loss.item():.6f}, "
                f"training_loss: {training_loss.item():.6f}, "
                f"img_rec_loss: {img_rec_loss.item():.6f}"
            )
            logfile.write(message + '\n')
            print(message)

            if opts.sample_freq > 0 and batches_done % opts.sample_freq == 0:
                img_sample = torch.cat((input_image.data, output_image.data, target_image.data), -2)
                save_file = os.path.join(sample_dir, f"train_epoch_{epoch}_batch_{batches_done}.png")
                save_image(img_sample, save_file, nrow=8, normalize=True)

            if opts.val_freq > 0 and batches_done % opts.val_freq == 0:
                val_loss_value = 0.0
                with torch.no_grad():
                    for val_idx, val_data in enumerate(val_loader):
                        if val_idx >= 20:
                            break
                        val_input_image = val_data['rendered'].to(device)
                        val_target_image = val_input_image.detach().clone()
                        val_target_clss = val_data['class'].to(device)
                        val_output = model(val_input_image, val_target_clss)

                        val_output_image = val_output['dec_out']
                        val_img_rec_loss = val_output['img_rec_loss'].mean()

                        val_loss_value = val_img_rec_loss.item()

                        val_img_sample = torch.cat((val_input_image.data, val_output_image.data, val_target_image.data), -2)
                        val_save_file = os.path.join(sample_dir, f"val_epoch_{epoch}_batch_{batches_done}.png")
                        save_image(val_img_sample, val_save_file, nrow=8, normalize=True)

                    val_loss_value /= 20
                    val_msg = (
                        f"Epoch: {epoch}/{opts.n_epochs}, Batch: {idx}/{len(train_loader)}, "
                        f"Imag mse loss: {val_loss_value: .6f}"
                    )
                    val_logfile.write(val_msg + "\n")
                    print(val_msg)

        if epoch % opts.ckpt_freq == 0:
            model_file = os.path.join(ckpt_dir, f"{opts.model_name}_{epoch}.pth")
            torch.save(model.module.state_dict(), model_file)

    logfile.close()
    val_logfile.close()


def train_svg_decoder(opts):
    pass


def train(opts):
    if opts.model_name == 'image_vae':
        train_image_vae(opts)
    elif opts.model_name == 'svg_decoder':
        train_svg_decoder(opts)
    else:
        raise NotImplementedError


def test(opts):
    pass


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
