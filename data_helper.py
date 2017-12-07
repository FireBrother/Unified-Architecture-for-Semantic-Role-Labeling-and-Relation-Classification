import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import json
import numpy as np


class SRLDataSet(Dataset):
    def __init__(self, filepath, word_dict_path, pos_dict_path, label_dict_path, max_len=250, is_test=False):
        _word_set = json.load(open(word_dict_path)).keys()
        self.word2idx = {w: i+1 for i, w in enumerate(_word_set)}
        self.word2idx[u'<UNK>'] = 0
        _pos_set = json.load(open(pos_dict_path)).keys()
        self.pos2idx = {w: i+1 for i, w in enumerate(_pos_set)}
        self.pos2idx[u'<UNK>'] = 0
        _label_set = json.load(open(label_dict_path)).keys()
        self.label2idx = {w: i for i, w in enumerate(_label_set)}
        self.is_test = is_test
        self.max_len = max_len

        self.data = []

        for lineno, line in enumerate(open(filepath, 'r', encoding='utf8')):
            tokens = line.strip().split()
            for token in tokens[:self.max_len]:
                v = token.split('/')
                if not self.is_test:
                    assert len(v) == 3, 'wrong length of token at line:{}-{} {}'.format(filepath, lineno, line)
                else:
                    assert len(v) == 3 or len(v) == 2,\
                        'wrong length of token at line:{}-{} {}'.format(filepath, lineno, line)
            self.data.append(tokens[:self.max_len])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        tokens = self.data[index]
        word_seq = []
        pos_seq = []
        rel_pos = -1
        output_seq = []
        for i, token in enumerate(tokens):
            v = token.split('/')
            word_seq.append(self.word2idx.get(v[0], 0))
            pos_seq.append(self.pos2idx.get(v[1], 0))
            if not self.is_test:
                output_seq.append(self.label2idx.get(v[2], 0))
            if len(v) == 3 and v[2] == 'rel':
                rel_pos = i
        sent_len = len(word_seq)
        word_seq = word_seq[:self.max_len]
        word_seq = np.pad(word_seq, (0, self.max_len-len(word_seq)), 'constant')
        pos_seq = pos_seq[:self.max_len]
        pos_seq = np.pad(pos_seq, (0, self.max_len - len(pos_seq)), 'constant')
        assert rel_pos != -1, 'No REL label found at line:{}-{}.\nMaybe max_len is too small.'.format(index, tokens)
        ret = {'word_seq': word_seq, 'pos_seq': pos_seq, 'rel_pos': rel_pos, 'sent_len': sent_len}
        if not self.is_test:
            output_seq = output_seq[:self.max_len]
            output_seq = np.pad(output_seq, (0, self.max_len - len(output_seq)), 'constant')
            ret['output_seq'] = output_seq
        return ret


if __name__ == '__main__':
    dataset = SRLDataSet('SRL_data/data/cpbdev.txt', 'SRL_data/data/word_dict.json', 'SRL_data/data/pos_dict.json',
                         'SRL_data/data/label_dict.json', is_test=False)
    dataloader = DataLoader(dataset=dataset, batch_size=2)
    for d in dataloader:
        print(d)
        break
