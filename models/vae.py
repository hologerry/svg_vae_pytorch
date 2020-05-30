from abc import abstractmethod
from typing import Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


"""ConditionalVAE
Implementation from https://github.com/AntixK/PyTorch-VAE
"""


class BaseVAE(nn.Module):
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        raise RuntimeWarning()

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass


class ConditionalVAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 img_size: int = 64,
                 kl_beta: float = 1.0,
                 **kwargs) -> None:
        super(ConditionalVAE, self).__init__()

        self.latent_dim = latent_dim
        self.img_size = img_size

        self.kl_beta = kl_beta

        self.embed_class = nn.Linear(num_classes, img_size * img_size)
        self.embed_data = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        in_channels += 1  # To account for the extra label channel
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(nn.Conv2d(in_channels, out_channels=h_dim,
                                                   kernel_size=3, stride=2, padding=1),
                                         nn.BatchNorm2d(h_dim),
                                         nn.ReLU()))
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)
        # self.fc_z = nn.Linear(hidden_dims[-1]*4, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim + num_classes, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Sequential(nn.ConvTranspose2d(hidden_dims[i],
                                                            hidden_dims[i + 1],
                                                            kernel_size=3,
                                                            stride=2,
                                                            padding=1,
                                                            output_padding=1),
                                         nn.BatchNorm2d(hidden_dims[i + 1]),
                                         nn.ReLU()))

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(nn.ConvTranspose2d(hidden_dims[-1],
                                                            hidden_dims[-1],
                                                            kernel_size=3,
                                                            stride=2,
                                                            padding=1,
                                                            output_padding=1),
                                         nn.BatchNorm2d(hidden_dims[-1]),
                                         nn.ReLU(),
                                         nn.Conv2d(hidden_dims[-1], out_channels=3,
                                                   kernel_size=3, padding=1),
                                         nn.Tanh())

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        assert not torch.isnan(input).any()
        result = self.encoder(input)
        assert not torch.isnan(result).any()
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        # z = self.fc_z(result)

        return [mu, log_var]
        # return z

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, label: Tensor) -> List[Tensor]:
        y = label.float()
        # print(y)
        assert not torch.isnan(y).any()
        # print("embed_class weight", self.embed_class.weight)
        # print("embed_class bias", self.embed_class.bias)
        embedded_class = self.embed_class(y)
        # print(embedded_class)
        assert not torch.isnan(embedded_class).any()
        embedded_class = embedded_class.view(-1, self.img_size, self.img_size).unsqueeze(1)
        embedded_input = self.embed_data(input)
        assert not torch.isnan(embedded_input).any()

        x = torch.cat([embedded_input, embedded_class], dim=1)
        mu, log_var = self.encode(x)
        # z = self.encode(x)

        z = self.reparameterize(mu, log_var)
        z = torch.cat([z, y], dim=1)
        return [self.decode(z), input, z, mu, log_var]
        # return [self.decode(z), input, z]

    def loss_function(self,
                      recons,
                      input,
                      z,
                      mu,
                      log_var,
                      **kwargs) -> dict:
        # recons = args[0]
        # input = args[1]
        # mu = args[2]
        # log_var = args[3]

        kld_weight = self.kl_beta  # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)
        assert not torch.isnan(mu).any()
        # print("mu max", torch.max(mu))
        assert not torch.isnan(log_var).any()
        mu2 = mu ** 2
        assert not torch.isnan(mu2).any()
        # print("mu2 max", torch.max(mu2))
        # print("logvar max", torch.max(log_var))
        log_var_exp = log_var.exp()
        assert not torch.isnan(log_var_exp).any()
        # print("log var exp max", torch.max(log_var_exp))
        term1 = 1 + log_var - mu2 - log_var_exp
        assert not torch.isnan(term1).any()
        # print("term1 max", torch.max(term1))
        kld_loss = torch.mean(-0.5 * torch.sum(term1, dim=1), dim=0)
        # kld_loss = torch.mean(-0.5 * torch.mean(term1, dim=1), dim=0)
        # print("kld_loss max", torch.max(kld_loss))
        assert not torch.isnan(kld_loss).any()

        loss = recons_loss + kld_weight * kld_loss
        # loss = recons_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': kld_loss}  # KLD shoud be -kld_loss, for convenience
        # return {'loss': recons_loss, 'Reconstruction_Loss': recons_loss, 'KLD': torch.tensor(0.0, device=input.device)}

    def sample(self,
               num_samples: int,
               current_device: int,
               labels: Tensor,
               **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        y = labels.float()
        z = torch.randn(num_samples, self.latent_dim)

        z = z.to(current_device)

        z = torch.cat([z, y], dim=1)
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x, **kwargs)[0]


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image = torch.randn((2, 1, 64, 64)).to(device)
    label = torch.ones((2, 1), dtype=torch.long).to(device)
    label = F.one_hot(label, num_classes=52).squeeze(dim=1)

    cvae = ConditionalVAE(1, 52, 32)
    out = cvae(image, label)
    output_img = out[0]
    print(output_img.size())
    losses = cvae.loss_function(*out)
    for k, v in losses.items():
        print(k, v)
