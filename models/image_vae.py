import torch
import torch.nn as nn
import torch.nn.functional as F

# from torch.distributions.bernoulli import Bernoulli
# from torch.distributions.independent import Independent

from models.layers import ConvCINReLu, UpsamplingConv


class VisualEncoder(nn.Module):
    def __init__(self, input_channels, base_depth, bottleneck_bits, num_categories):
        super(VisualEncoder, self).__init__()
        self.bottleneck_bits = bottleneck_bits
        self.down1 = ConvCINReLu(inch=input_channels, outch=base_depth, kernel_size=5, stride=1, num_categories=num_categories)
        self.down2 = ConvCINReLu(inch=base_depth, outch=base_depth, kernel_size=5, stride=2, num_categories=num_categories)
        self.down3 = ConvCINReLu(inch=base_depth, outch=2 * base_depth, kernel_size=5, stride=1, num_categories=num_categories)
        self.down4 = ConvCINReLu(inch=2 * base_depth, outch=2 * base_depth, kernel_size=5, stride=2, num_categories=num_categories)
        self.down5 = ConvCINReLu(inch=2 * base_depth, outch=2 * base_depth, kernel_size=4, stride=2, num_categories=num_categories)
        self.down6 = ConvCINReLu(inch=2 * base_depth, outch=2 * base_depth, kernel_size=4, stride=2, num_categories=num_categories)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(1024, 2 * bottleneck_bits, bias=True)

    def forward(self, x, clss):
        out = self.down1(x, clss)
        out = self.down2(out, clss)
        out = self.down3(out, clss)
        out = self.down4(out, clss)
        out = self.down5(out, clss)
        out = self.down6(out, clss)
        out = self.flatten(out)
        out = self.fc(out)
        return out


# class Bottleneck(nn.Module):
#     def __init__(self, bottleneck_bits, free_bits, kl_beta, mode='Train'):
#         super(Bottleneck, self).__init__()
#         self.z_size = bottleneck_bits
#         self.free_bits = free_bits
#         self.kl_beta = kl_beta
#         self.mode = mode

#     def forward(self, x):
#         mu = x[..., :self.z_size]
#         x_shape = list(x.size())

#         if self.mode != 'train':
#             return {'z': mu, 'b_loss': torch.tensor(0.0, device=x.device)}

#         log_sigma = x[..., self.z_size:]
#         epsilon = torch.randn(x_shape[:-1] + [self.z_size], device=x.device)
#         z = mu + torch.exp(log_sigma / 2) * epsilon
#         kl = 0.5 * torch.mean(torch.exp(log_sigma) + mu ** 2 - 1.0 - log_sigma, dim=-1)
#         zero = torch.zeros_like(kl)
#         kl_loss = torch.mean(torch.max(kl - self.free_bits, zero))
#         output = {}
#         output['z'] = z
#         output['b_loss'] = torch.mean(kl_loss * self.kl_beta)

#         return output


# class VisualDecoder(nn.Module):
#     def __init__(self, base_depth, bottleneck_bits, output_channels, num_categories):
#         super(VisualDecoder, self).__init__()
#         self.fc = nn.Linear(bottleneck_bits, 1024)

#         self.up1 = UpsamplingConv(2 * base_depth, 2 * base_depth, kernel_size=4, stride=2, num_categories=num_categories)
#         self.up2 = UpsamplingConv(2 * base_depth, 2 * base_depth, kernel_size=4, stride=2, num_categories=num_categories)
#         self.up3 = UpsamplingConv(2 * base_depth, 2 * base_depth, kernel_size=5, stride=1, num_categories=num_categories)
#         self.up4 = UpsamplingConv(2 * base_depth, 2 * base_depth, kernel_size=5, stride=2, num_categories=num_categories)
#         self.up5 = UpsamplingConv(2 * base_depth, base_depth, kernel_size=5, stride=1, num_categories=num_categories)
#         self.up6 = UpsamplingConv(base_depth, base_depth, kernel_size=5, stride=2, num_categories=num_categories)
#         self.up7 = UpsamplingConv(base_depth, base_depth, kernel_size=5, stride=1, num_categories=num_categories)

#         self.conv = nn.Conv2d(base_depth, output_channels, kernel_size=5, padding=2)

#     def forward(self, bottleneck, clss):
#         out = self.fc(bottleneck)
#         out = out.view([-1, 64, 4, 4])
#         out = self.up1(out, clss)
#         out = self.up2(out, clss)
#         out = self.up3(out, clss)
#         out = self.up4(out, clss)
#         out = self.up5(out, clss)
#         out = self.up6(out, clss)
#         out = self.up7(out, clss)
#         out = self.conv(out)
#         ber = Bernoulli(logits=out)
#         out = Independent(ber, reinterpreted_batch_ndims=3)
#         return out


class ImageVAE(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, num_categories=62,
                 base_depth=32, bottleneck_bits=32, free_bits=0.15,
                 kl_beta=300, mode='train'):
        super(ImageVAE, self).__init__()
        self.mode = mode
        self.kl_beta = kl_beta
        self.bottleneck_bits = bottleneck_bits
        self.visual_encoder = VisualEncoder(input_channels, base_depth, bottleneck_bits, num_categories)
        # self.bottleneck = Bottleneck(bottleneck_bits, free_bits, kl_beta, mode)
        # self.visual_decoder = VisualDecoder(base_depth, bottleneck_bits, output_channels, num_categories)
        # becuase multigpu output must be tensor or dict or tensor, decoder output is distribution
        self.fc = nn.Linear(bottleneck_bits, 1024)
        self.up1 = UpsamplingConv(2 * base_depth, 2 * base_depth, kernel_size=4, stride=2, num_categories=num_categories)  # 8
        self.up2 = UpsamplingConv(2 * base_depth, 2 * base_depth, kernel_size=4, stride=2, num_categories=num_categories)  # 16
        self.up3 = UpsamplingConv(2 * base_depth, 2 * base_depth, kernel_size=5, stride=1, num_categories=num_categories)  # 16
        self.up4 = UpsamplingConv(2 * base_depth, 2 * base_depth, kernel_size=5, stride=2, num_categories=num_categories)  # 32
        self.up5 = UpsamplingConv(2 * base_depth, base_depth, kernel_size=5, stride=1, num_categories=num_categories)  # 32
        self.up6 = UpsamplingConv(base_depth, base_depth, kernel_size=5, stride=2, num_categories=num_categories)  # 64
        self.up7 = UpsamplingConv(base_depth, base_depth, kernel_size=5, stride=1, num_categories=num_categories)  # 64
        self.conv = nn.Conv2d(base_depth, output_channels, kernel_size=5, padding=2)  # 64

        self.sigmoid = nn.Sigmoid()
        # self.rec_criterion = nn.BCELoss()

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size(), device=mu.device)
        z = mu + std * esp
        return z

    def forward(self, inputs, clss):
        enc_out = self.visual_encoder(inputs, clss)
        enc_out = enc_out.view(-1, 2 * self.bottleneck_bits)
        # b_output = self.bottleneck(enc_out)
        # sampled_bottleneck, b_loss = b_output['z'], b_output['b_loss']

        mu, logvar = torch.chunk(enc_out, 2, dim=-1)
        z = self.reparameterize(mu, logvar)

        dec_out = self.fc(z)
        dec_out = dec_out.view([-1, 64, 4, 4])
        dec_out = self.up1(dec_out, clss)
        dec_out = self.up2(dec_out, clss)
        dec_out = self.up3(dec_out, clss)
        dec_out = self.up4(dec_out, clss)
        dec_out = self.up5(dec_out, clss)
        dec_out = self.up6(dec_out, clss)
        dec_out = self.up7(dec_out, clss)
        dec_out = self.conv(dec_out)
        dec_out = self.sigmoid(dec_out)
        # ber = Bernoulli(logits=dec_out)
        # dec_out = Independent(ber, reinterpreted_batch_ndims=3)
        # output_img = dec_out.mean

        output = {}
        output['dec_out'] = dec_out
        output['samp_b'] = z

        if self.mode == 'train':
            # calculating loss
            output['b_loss'] = self.kl_beta * (-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp()))
            # rec_loss = -dec_out.log_prob(inputs)
            # elbo = torch.mean(-(b_loss + rec_loss))
            # rec_loss = torch.mean(rec_loss)
            # training_loss = -elbo
            # output['rec_loss'] = rec_loss
            # output['training_loss'] = training_loss
            output['rec_loss'] = F.binary_cross_entropy(dec_out, inputs, size_average=False)
        # dec_out = self.visual_decoder(sampled_bottleneck, clss)
        return output


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image = torch.randn((2, 1, 64, 64)).to(device)
    clss = torch.ones((2, 1), dtype=torch.long).to(device)

    image_vae = ImageVAE().to(device)
    output = image_vae(image, clss)
    dec_out = output['dec_out']
    sampled_bottleneck = output['samp_b']
    b_loss = output['b_loss']
    rec_loss = output['rec_loss']
    # training_loss = output['training_loss']
    print(dec_out.size())
    print(sampled_bottleneck.size())
    print(b_loss.size())
    print(rec_loss.size())
    # print(training_loss.size())
