import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.image_vae import ImageVAE


def shift_right(x, pad_value=None):
    if pad_value is None:
        # the pad arg is move from last dim to first dim
        shifted = torch.pad(x, (0, 0, 0, 0, 0, 0, 1, 0))[:-1, :, :, :]
    else:
        shifted = torch.cat([pad_value, x], axis=0)[:-1, :, :, :]
    return shifted


def length_form_embedding(emb):
    """Compute the length of each sequence in the batch
    Args:
        emb: [seq_len, batch, depth]
    Returns:
        a 0/1 tensor: [batch]
    """
    absed = torch.abs(emb)
    sum_last = torch.sum(absed, dim=2, keepdim=True)
    mask = sum_last != 0
    sum_except_batch = torch.sum(mask, dim=(0, 2), dtype=torch.long)
    return sum_except_batch


class SVGDecoder(nn.Module):
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
        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, self.num_hidden_layers, dropout=dropout_p)
        self.dropout = nn.Dropout(dropout_p)

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
        output, (hidden, cell) = self.rnn(inpt, (hidden, cell))
        return output, (hidden, cell)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 2
    seq_len = 20
    feature_dim = 10  # like vocab_size
    bottleneck_bits = 32
    num_categories = 52
    hidden_size = 1024
    sg_bottleneck = True
    tearcher_force_ratio = 1.0

    image = torch.randn((batch_size, 1, 64, 64)).to(device)
    clss = torch.randint(low=0, high=num_categories, size=(batch_size, 1), dtype=torch.long).to(device)

    image_vae = ImageVAE().to(device)
    svg_decoder = SVGDecoder().to(device)
    if sg_bottleneck:
        image_vae = image_vae.eval().to(device)

    sampled_bottleneck, _, _ = image_vae(image, clss)

    trg = torch.randn((seq_len, batch_size, feature_dim)).to(device)  # [seq_len, batch_size, feature_dim]
    trg_len = trg.size(0)

    outputs = torch.zeros(trg_len, batch_size, hidden_size).to(device)

    inpt = trg[0, :, :]

    hidden, cell = svg_decoder.init_state_input(sampled_bottleneck)

    for t in range(1, trg_len):
        output, (hidden, cell) = svg_decoder(inpt, sampled_bottleneck, clss, hidden, cell)
        outputs[t] = output
        teacher_force = random.random() < tearcher_force_ratio
        topv, topi = output.topk(1)
        inpt = trg[t] if teacher_force else topi

    print(outputs.size())
