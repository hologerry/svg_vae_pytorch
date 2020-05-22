import os
import pickle

import torch
import torch.utils.data as data


class SVGDataset(data.Dataset):
    def __init__(self, root_path, max_seq_len=51, seq_feature_dim=10, mode='train'):
        super().__init__()
        self.mode = mode
        self.pkl_path = os.path.join(root_path, self.mode, f'{mode}_all.pkl')
        pkl_f = open(self.pkl_path, 'rb')
        print(f"Loading {self.pkl_path} pkl file ...")
        self.all_glyphs = pickle.load(pkl_f)
        pkl_f.close()
        print(f"Finished loading")
        self.max_seq_len = max_seq_len
        self.feature_dim = seq_feature_dim

    def __getitem__(self, index):
        cur_glyph = self.all_glyphs[index]
        item = {}
        item['class'] = torch.LongTensor(cur_glyph['class'])
        item['seq_len'] = torch.LongTensor(cur_glyph['seq_len'])
        item['sequence'] = torch.FloatTensor(cur_glyph['sequence']).view(self.max_seq_len, self.feature_dim)
        item['rendered'] = torch.FloatTensor(cur_glyph['rendered']).view(1, 64, 64)
        return item

    def __len__(self):
        return len(self.all_glyphs)


def get_loader(root_path, max_seq_len, seq_feature_dim, batch_size, mode='train'):
    dataset = SVGDataset(root_path, max_seq_len, seq_feature_dim, mode)
    dataloader = data.DataLoader(dataset, batch_size, shuffle=(mode == 'train'), num_workers=batch_size)

    return dataloader


if __name__ == '__main__':
    root_path = 'svg_vae_data/glyph_pkl_dataset'
    max_seq_len = 51
    seq_feature_dim = 10
    batch_size = 2

    loader = get_loader(root_path, max_seq_len, seq_feature_dim, batch_size, 'test')

    for idx, batch in enumerate(loader):
        if idx > 0:
            break
        print('class', batch['class'].size())
        print('seq_len', batch['seq_len'].size())
        print('sequence', batch['sequence'].size())
        print('rendered', batch['rendered'].size())
        print(torch.max(batch['rendered']))
