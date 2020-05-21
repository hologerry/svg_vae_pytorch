import torch


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


def lognormal(y, mean, logstd, logsqrttwopi):
    return -0.5 * (y - mean) / torch.exp(logstd) ** 2 - logstd - logsqrttwopi
