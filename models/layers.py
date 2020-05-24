import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F


"""
These code is from:
https://github.com/ChinmayLad/neural-style-transfer
"""


class ConditionalInstanceNorm(nn.Module):
    """
    PyTorch does not contain api for Conditional Instance Normalization.
    This is a implementation that uses code from torch BatchNorm and is
    tweaked as to have condition learnable weights.
    [link](https://pytorch.org/docs/stable/_modules/torch/nn/modules/batchnorm.html)
    The idea is to have weight tensor of size L x C, where L is the no. of
    style representations during training.
    During the forward pass we provide input image as well as style condition "label" ~ [0,L).
    During backward pass only weigth[label:] tensor are updated and rest remaing the same.
    Here weight and bias refer to \gamma and \beta used for normalization.  # noqa
    """

    _version = 1
    __constants__ = ['track_running_stats', 'momentum', 'eps', 'weight', 'bias',
                     'running_mean', 'running_var', 'num_batches_tracked',
                     'num_features', 'affine']

    def __init__(self, num_features, num_labels, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(ConditionalInstanceNorm, self).__init__()
        self.num_features = num_features
        self.num_labels = num_labels
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_labels, num_features))
            self.bias = Parameter(torch.Tensor(num_labels, num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def _check_input_dim(self, inpt):
        raise NotImplementedError

    def forward(self, inpt, label):
        # self._check_input_dim(inpt)
        ins = F.instance_norm(
            inpt, self.running_mean, self.running_var, None, None,
            self.training or not self.track_running_stats,
            self.momentum, self.eps)
        if torch.max(label) >= self.num_labels:
            raise ValueError('Expected label to be < than {} but got {}'.format(self.num_labels, label))
        w = self.weight
        b = self.bias
        if self.affine:
            w = self.weight[label].view(inpt.size(0), self.num_features).unsqueeze(2).unsqueeze(3)
            b = self.bias[label].view(inpt.size(0), self.num_features).unsqueeze(2).unsqueeze(3)
            return ins * w + b
        else:
            return ins

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)

        if (version is None or version < 1) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + 'num_batches_tracked'
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super(ConditionalInstanceNorm, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)


class ConvCINReLu(nn.Module):
    """
    A custom convolution layer that contains reflection_padding, conditional_instance_norm
    and a relu activation.
    The forward pass takes in label as an input that is used in normalization.
    """
    def __init__(self, inch, outch, kernel_size, stride, num_categories, activation='relu'):
        super(ConvCINReLu, self).__init__()
        padding = kernel_size // 2
        if kernel_size % 2 == 1:
            self.reflection = nn.ReflectionPad2d(padding)
        else:
            self.reflection = nn.ReflectionPad2d((padding - 1, padding, padding - 1, padding))
        self.conv = nn.Conv2d(inch, outch, kernel_size, stride, padding=0)
        # self.norm = ConditionalInstanceNorm(outch, num_categories)
        self.norm = nn.InstanceNorm2d(outch)
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'leaky':
            self.act = nn.LeakyReLU(0.02)

    def forward(self, x, label):
        x = self.reflection(x)
        x = self.conv(x)
        x = self.norm(x, label)
        x = self.act(x)
        return x


class UpsamplingConv(nn.Module):
    def __init__(self, inch, outch, kernel_size, stride, num_categories, activation='relu'):
        super(UpsamplingConv, self).__init__()
        # self.upsample = nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=False)
        # self.conv = ConvCINReLu(inch, outch, kernel_size, stride=1, num_categories=num_categories)
        pad = (kernel_size - 1) // 2
        out_pad = 1 if kernel_size % 2 == 1 and stride % 2 == 0 else 0
        # if kernel_size == 4 and stride == 2:
        #     pad = 1
        #     out_pad = 0
        # elif kernel_size == 5 and stride == 2:
        #     pad = 2
        #     out_pad = 1
        # elif kernel_size == 5 and stride == 1:
        #     pad = 2
        #     out_pad = 0
        self.deconv = nn.ConvTranspose2d(inch, outch, kernel_size, stride, padding=pad, output_padding=out_pad)
        # self.norm = ConditionalInstanceNorm(outch, num_categories)
        self.norm = nn.InstanceNorm2d(outch)
        self.act = nn.ReLU()

    def forward(self, x, label):
        # out = self.upsample(x)
        # out = self.conv(out, label)
        out = self.deconv(x)
        out = self.norm(out, label)
        out = self.act(out)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, inch, outch, kernel_size, num_categories, activation='relu'):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvCINReLu(inch, outch, kernel_size, stride=1, num_categories=num_categories)
        self.conv2 = ConvCINReLu(outch, inch, kernel_size, stride=1, num_categories=num_categories)

    def forward(self, x, label):
        h1 = self.conv1(x, label)
        h2 = self.conv2(h1, label)
        return x + h2
