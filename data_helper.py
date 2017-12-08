import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import json
import numpy as np
from lca import Tree

class SRLDataSet(Dataset):
    def __init__(self, filepath, word_dict_path, pos_dict_path, label_dict_path ,depend_dict_path,tree_path, max_len=250,max_depend_len = 20, is_test=False):
        _word_set = json.load(open(word_dict_path)).keys()
        self.word2idx = {w: i+1 for i, w in enumerate(_word_set)}
        self.word2idx[u'<UNK>'] = 0
        _pos_set = json.load(open(pos_dict_path)).keys()
        self.pos2idx = {w: i+1 for i, w in enumerate(_pos_set)}
        self.pos2idx[u'<UNK>'] = 0
        _depend_set = json.load(open(depend_dict_path)).keys()
        self.depend2idx = {w: i+1 for i, w in enumerate(_depend_set)}
        self.depend2idx[u'<UNK>'] = 0
        _label_set = json.load(open(label_dict_path)).keys()
        self.label2idx = {w: i for i, w in enumerate(_label_set)}
        self.is_test = is_test
        self.max_len = max_len
        self.max_depend_len = max_depend_len
        self.data = []
        self.trees = []
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

        for line in open(tree_path, 'r', encoding='utf8'):
            self.trees.append(line.strip())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        tokens = self.data[index]
        node_list = self.trees[index]

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
        tree = Tree(node_list,rel_pos)
        token_path = []
        depend_path = []
        rel_token_path = []
        rel_depend_path = []
        path_len = []
        rel_path_len = []
        for i in range(1,tree.size):
            if not i == rel_pos:
                tmp = tree.cal_lca(i)
                print(tmp)
                path_len.append(len(tmp[0]))
                rel_path_len.append(len(tmp[1]))
                token_path.append(np.pad(list(map(lambda x: word_seq[x], tmp[0])),(0,self.max_depend_len-len(tmp[0])),'constant'))
                depend_path.append(np.pad(list(map(lambda x: self.depend2idx[x], tmp[1])),(0,self.max_depend_len-len(tmp[1])),'constant'))
                rel_token_path.append(np.pad(list(map(lambda x: word_seq[x], tmp[2])),(0,self.max_depend_len-len(tmp[2])),'constant'))
                rel_depend_path.append(np.pad(list(map(lambda x: self.depend2idx[x], tmp[3])),(0,self.max_depend_len-len(tmp[3])),'constant'))
        path_len = np.pad(path_len, (0, self.max_len - len(path_len)), 'constant')
        rel_path_len = np.pad(rel_path_len, (0, self.max_len - len(rel_path_len)), 'constant')

        ret = {'word_seq': word_seq, 'pos_seq': pos_seq, 'rel_pos': rel_pos, 'sent_len': sent_len,
               'token_path':np.asarray(token_path),'depend_path':np.asarray(depend_path),
               'rel_token_path':np.asarray(rel_token_path),'rel_depend_path':np.asarray(rel_depend_path),
               'path_len':path_len,'rel_path_len' : rel_path_len
               }
        if not self.is_test:
            output_seq = output_seq[:self.max_len]
            output_seq = np.pad(output_seq, (0, self.max_len - len(output_seq)), 'constant')
            ret['output_seq'] = output_seq
        return ret


if __name__ == '__main__':
    dataset = SRLDataSet('SRL_data/data/cpbdev.txt', 'SRL_data/data/word_dict.json', 'SRL_data/data/pos_dict.json',
                         'SRL_data/data/label_dict.json','SRL_data/data/depend_dict.json','SRL_data/data/cpbdev_tree.txt', is_test=False)
    dataloader = DataLoader(dataset=dataset, batch_size=1)
    for d in dataloader:
        print(d['path_len'])
        break
