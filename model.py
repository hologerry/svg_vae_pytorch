import torch
from torch import nn
from layers import ConvCINReLu, UpsamplingConv
from torch.distributions.independent import Independent
from torch.distributions.bernoulli import Bernoulli


class VisualEncoder(nn.Module):
    def __init__(self, input_channels, base_depth, bottleneck_bits, num_categories):
        super(VisualEncoder, self).__init__()
        self.bottleneck_bits = bottleneck_bits
        self.down1 = ConvCINReLu(inch=input_channels, outch=base_depth, kernel_size=5, stride=1, labels=num_categories)
        self.down2 = ConvCINReLu(inch=base_depth, outch=base_depth, kernel_size=5, stride=2, labels=num_categories)
        self.down3 = ConvCINReLu(inch=base_depth, outch=2*base_depth, kernel_size=5, stride=1, labels=num_categories)
        self.down4 = ConvCINReLu(inch=2*base_depth, outch=2*base_depth, kernel_size=5, stride=2, labels=num_categories)
        self.down5 = ConvCINReLu(inch=2*base_depth, outch=2*base_depth, kernel_size=4, stride=2, labels=num_categories)
        self.down6 = ConvCINReLu(inch=2*base_depth, outch=2*base_depth, kernel_size=4, stride=2, labels=num_categories)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(1024, 2 * bottleneck_bits, bias=True)

    def forward(self, x, label):
        out = self.down1(x, label)
        out = self.down2(out, label)
        out = self.down3(out, label)
        out = self.down4(out, label)
        out = self.down5(out, label)
        out = self.down6(out, label)
        out = torch.reshape(out, [-1, 2 * self.bottleneck_bits])
        return out


class Bottleneck(nn.Module):
    def __init__(self, bottleneck_bits, free_bits, kl_lambda, mode):
        super(Bottleneck, self).__init__()
        self.z_size = bottleneck_bits
        self.free_bits = free_bits
        self.kl_lambda = kl_lambda
        self.mode = mode

    def forward(self, x):
        mu = x[..., :self.z_size]
        x_shape = x.size()
        if self.mode != 'train':
            return mu, 0.0
        log_sigma = x[..., self.z_size:]
        epsilon = torch.randn(x_shape[:-1] + [self.z_size])
        z = mu + torch.exp(log_sigma / 2) * epsilon
        kl = 0.5 * torch.mean(torch.exp(log_sigma) + torch.pow(mu, 2) - 1.0 - log_sigma, dim=-1)
        zero = torch.zeros_like(kl)
        kl_loss = torch.mean(torch.max(kl - self.free_bits, zero))
        return z, kl_loss * self.kl_lambda


class VisualDecoder(nn.Module):
    def __init__(self, base_depth, bottleneck_bits, output_channels, num_categories):
        super(VisualDecoder, self).__init__()
        self.fc = nn.Linear(bottleneck_bits, 1024)

        self.up1 = UpsamplingConv(2*base_depth, 2*base_depth, kernel_size=4, stride=2, labels=num_categories)
        self.up2 = UpsamplingConv(2*base_depth, 2*base_depth, kernel_size=4, stride=2, labels=num_categories)
        self.up3 = UpsamplingConv(2*base_depth, 2*base_depth, kernel_size=5, stride=1, labels=num_categories)
        self.up4 = UpsamplingConv(2*base_depth, 2*base_depth, kernel_size=5, stride=2, labels=num_categories)
        self.up5 = UpsamplingConv(2*base_depth, base_depth, kernel_size=5, stride=1, labels=num_categories)
        self.up6 = UpsamplingConv(base_depth, base_depth, kernel_size=5, stride=2, labels=num_categories)
        self.up7 = UpsamplingConv(base_depth, base_depth, kernel_size=5, stride=1, labels=num_categories)

        self.conv = nn.Conv2D(base_depth, output_channels)

    def forward(self, bottleneck, label):
        out = self.fc(bottleneck)
        out = self.up1(out)
        out = self.up2(out)
        out = self.up3(out)
        out = self.up4(out)
        out = self.up5(out)
        out = self.up6(out)
        out = self.up7(out)
        out = self.conv(out)
        ber = Bernoulli(logits=out)
        out = Independent(ber, reinterpreted_batch_ndims=3)
        return out


class ImageVAE(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, num_categories=62,
                 base_depth=32, bottleneck_bits=32, free_bits=0.15,
                 kl_lambda=300, mode='train'):
        super(ImageVAE, self).__init__()
        self.mode = mode
        self.bottleneck_bits = bottleneck_bits
        self.visual_encoder = VisualEncoder(input_channels)
        self.bottleneck = Bottleneck(bottleneck_bits, free_bits, kl_lambda, mode)
        self.visual_decoder = VisualDecoder(base_depth, bottleneck_bits, output_channels, num_categories)

    def forward(self, features):
        inputs, targets = features['inputs'], features['targets']
        enc_out = self.visual_encoder(inputs, targets)
        enc_out = torch.reshape(enc_out, (-1, 2 * self.bottleneck))
        sample_bottleneck, b_loss = self.bottleneck(enc_out)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image = torch.randn((2, 3, 64, 64)).to(device)
    image_vae = ImageVAE()
