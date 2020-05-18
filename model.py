import torch
from torch import nn
from layers import ConvCINReLu, UpsamplingConv
from torch.distributions.independent import Independent
from torch.distributions.bernoulli import Bernoulli


class VisualEncoder(nn.Module):
    def __init__(self, input_channels, base_depth, bottleneck_bits, num_categories):
        super(VisualEncoder, self).__init__()
        self.bottleneck_bits = bottleneck_bits
        self.down1 = ConvCINReLu(inch=input_channels, outch=base_depth, kernel_size=5, stride=1, num_categories=num_categories)
        self.down2 = ConvCINReLu(inch=base_depth, outch=base_depth, kernel_size=5, stride=2, num_categories=num_categories)
        self.down3 = ConvCINReLu(inch=base_depth, outch=2*base_depth, kernel_size=5, stride=1, num_categories=num_categories)
        self.down4 = ConvCINReLu(inch=2*base_depth, outch=2*base_depth, kernel_size=5, stride=2, num_categories=num_categories)
        self.down5 = ConvCINReLu(inch=2*base_depth, outch=2*base_depth, kernel_size=4, stride=2, num_categories=num_categories)
        self.down6 = ConvCINReLu(inch=2*base_depth, outch=2*base_depth, kernel_size=4, stride=2, num_categories=num_categories)
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


class Bottleneck(nn.Module):
    def __init__(self, bottleneck_bits, free_bits, kl_beta, mode='Train'):
        super(Bottleneck, self).__init__()
        self.z_size = bottleneck_bits
        self.free_bits = free_bits
        self.kl_beta = kl_beta
        self.mode = mode

    def forward(self, x):
        mu = x[..., :self.z_size]
        x_shape = list(x.size())

        if self.mode != 'train':
            return mu, 0.0

        log_sigma = x[..., self.z_size:]
        epsilon = torch.randn(x_shape[:-1] + [self.z_size])
        z = mu + torch.exp(log_sigma / 2) * epsilon
        kl = 0.5 * torch.mean(torch.exp(log_sigma) + torch.pow(mu, 2) - 1.0 - log_sigma, dim=-1)
        zero = torch.zeros_like(kl)
        kl_loss = torch.mean(torch.max(kl - self.free_bits, zero))
        return z, kl_loss * self.kl_beta


class VisualDecoder(nn.Module):
    def __init__(self, base_depth, bottleneck_bits, output_channels, num_categories):
        super(VisualDecoder, self).__init__()
        self.fc = nn.Linear(bottleneck_bits, 1024)

        self.up1 = UpsamplingConv(2*base_depth, 2*base_depth, kernel_size=4, stride=2, num_categories=num_categories)
        self.up2 = UpsamplingConv(2*base_depth, 2*base_depth, kernel_size=4, stride=2, num_categories=num_categories)
        self.up3 = UpsamplingConv(2*base_depth, 2*base_depth, kernel_size=5, stride=1, num_categories=num_categories)
        self.up4 = UpsamplingConv(2*base_depth, 2*base_depth, kernel_size=5, stride=2, num_categories=num_categories)
        self.up5 = UpsamplingConv(2*base_depth, base_depth, kernel_size=5, stride=1, num_categories=num_categories)
        self.up6 = UpsamplingConv(base_depth, base_depth, kernel_size=5, stride=2, num_categories=num_categories)
        self.up7 = UpsamplingConv(base_depth, base_depth, kernel_size=5, stride=1, num_categories=num_categories)

        self.conv = nn.Conv2d(base_depth, output_channels, kernel_size=5, padding=2)

    def forward(self, bottleneck, clss):
        out = self.fc(bottleneck)
        out = out.view([-1, 64, 4, 4])
        out = self.up1(out, clss)
        out = self.up2(out, clss)
        out = self.up3(out, clss)
        out = self.up4(out, clss)
        out = self.up5(out, clss)
        out = self.up6(out, clss)
        out = self.up7(out, clss)
        out = self.conv(out)
        ber = Bernoulli(logits=out)
        out = Independent(ber, reinterpreted_batch_ndims=3)
        return out


class ImageVAE(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, num_categories=62,
                 base_depth=32, bottleneck_bits=32, free_bits=0.15,
                 kl_beta=300, mode='train'):
        super(ImageVAE, self).__init__()
        self.mode = mode
        self.bottleneck_bits = bottleneck_bits
        self.visual_encoder = VisualEncoder(input_channels, base_depth, bottleneck_bits, num_categories)
        self.bottleneck = Bottleneck(bottleneck_bits, free_bits, kl_beta, mode)
        self.visual_decoder = VisualDecoder(base_depth, bottleneck_bits, output_channels, num_categories)

    def forward(self, inputs, clss):
        enc_out = self.visual_encoder(inputs, clss)
        enc_out = enc_out.view(-1, 2 * self.bottleneck_bits)
        sampled_bottleneck, b_loss = self.bottleneck(enc_out)
        dec_out = self.visual_decoder(sampled_bottleneck, clss)
        return dec_out, b_loss


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image = torch.randn((2, 1, 64, 64)).to(device)
    clss = torch.ones((2, 1), dtype=torch.long).to(device)

    image_vae = ImageVAE()
    output, bottleneck_loss = image_vae(image, clss)
    bottleneck_kl = torch.mean(bottleneck_loss)
    # calculating loss
    rec_loss = -output.log_prob(image)
    elbo = torch.mean(-(bottleneck_loss + rec_loss))
    rec_loss = torch.mean(rec_loss)
    training_loss = -elbo
    print(output.mean, bottleneck_kl, rec_loss, training_loss)
