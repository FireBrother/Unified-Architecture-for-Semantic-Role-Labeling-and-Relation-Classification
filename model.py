import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np

from data_helper import SRLDataSet


class UnifiedFramework(nn.Module):
    def __init__(self, config):
        super(UnifiedFramework, self).__init__()
        self.config = config
        self.pretrained_word_embedding = nn.Embedding(config['vocab_size'], config['embedding_dim'])
        self.untrained_word_embedding = nn.Embedding(config['vocab_size'], config['embedding_dim'])
        self.pos_embedding = nn.Embedding(config['pos_set_size'], config['embedding_dim'])
        self.depend_embedding = nn.Embedding(config['depend_set_size'], config['embedding_dim'])
        self.global_context_repr = nn.LSTM(config['feature_size'], config['gcr_hidden_size'],
                                           config['gcr_num_layers'], bidirectional=True, batch_first=True,
                                           dropout=config['drop_out'])
        self.generic_path_repr_forward = nn.LSTM(config['feature_size'], config['gpr_hidden_size'],
                                                 config['gpr_num_layers'], bidirectional=False, batch_first=True,
                                                 dropout=config['drop_out'])
        self.generic_path_repr_backward = nn.LSTM(config['feature_size'], config['gpr_hidden_size'],
                                                  config['gpr_num_layers'], bidirectional=False, batch_first=True,
                                                  dropout=config['drop_out'])
        self.relation_path_repr_forward = nn.LSTM(config['embedding_dim'], config['rpr_hidden_size'],
                                                  config['rpr_num_layers'], bidirectional=False, batch_first=True,
                                                  dropout=config['drop_out'])
        self.relation_path_repr_backward = nn.LSTM(config['embedding_dim'], config['rpr_hidden_size'],
                                                   config['rpr_num_layers'], bidirectional=False, batch_first=True,
                                                   dropout=config['drop_out'])
        self.word_lex_repr_weight = nn.Linear(config['embedding_dim'] * 3, config['feature_size'])
        self.context_repr_weight = nn.Linear(config['gcr_hidden_size'] * 4, config['feature_size'])
        self.path_repr_weight = nn.Linear(config['gpr_hidden_size'] * 2 + config['rpr_hidden_size'] * 2,
                                          config['feature_size'])
        self.output = nn.Sequential(
            nn.Linear(config['feature_size'], config['categories']),
            nn.Softmax(dim=-1)
        )

    def forward(self, word_seq, pos_seq, rel_pos, sent_len, token_index_path, depend_path,
                rel_token_index_path, rel_depend_path, path_len, rel_path_len):
        # sort the input sequences by length
        sent_len, sent_len_sort_idx = torch.sort(sent_len, descending=True)
        _, sent_len_unsort_idx = torch.sort(sent_len_sort_idx)

        word_seq = word_seq.index_select(0, sent_len_sort_idx)
        pos_seq = pos_seq.index_select(0, sent_len_sort_idx)
        token_index_path = token_index_path.index_select(0, sent_len_sort_idx)
        depend_path = depend_path.index_select(0, sent_len_sort_idx)
        rel_token_index_path = rel_token_index_path.index_select(0, sent_len_sort_idx)
        rel_depend_path = rel_depend_path.index_select(0, sent_len_sort_idx)
        path_len = path_len.index_select(0, sent_len_sort_idx)
        rel_path_len = rel_path_len.index_select(0, sent_len_sort_idx)

        # Global Context Representation
        word_lex_repr = self.word_repr(word_seq, pos_seq)
        packed_word_lex_repr = pack_padded_sequence(word_lex_repr, sent_len.data.numpy(), batch_first=True)
        hidden = self.init_hidden(len(sent_len), self.config['gcr_num_layers'] * 2, self.config['gcr_hidden_size'])
        packed_gcr, hidden = self.global_context_repr(packed_word_lex_repr, hidden)
        gcr, _ = pad_packed_sequence(packed_gcr, batch_first=True)

        # generate token and pos path by index path
        token_path = word_seq.unsqueeze(1).expand_as(token_index_path).gather(2, token_index_path)
        pos_path = pos_seq.unsqueeze(1).expand_as(token_index_path).gather(2, token_index_path)
        rel_token_path = word_seq.unsqueeze(1).expand_as(rel_token_index_path).gather(2, rel_token_index_path)
        rel_pos_path = pos_seq.unsqueeze(1).expand_as(rel_token_index_path).gather(2, rel_token_index_path)

        # sort the path sequences by length
        path_len = path_len.view(-1)
        rel_path_len = rel_path_len.view(-1)
        token_path = token_path.view(token_path.size(0) * token_path.size(1), token_path.size(2))
        pos_path = pos_path.view(pos_path.size(0) * pos_path.size(1), pos_path.size(2))
        depend_path = depend_path.view(depend_path.size(0) * depend_path.size(1), depend_path.size(2))
        rel_pos_path = rel_pos_path.view(rel_pos_path.size(0) * rel_pos_path.size(1), rel_pos_path.size(2))
        rel_token_path = rel_token_path.view(rel_token_path.size(0) * rel_token_path.size(1), rel_token_path.size(2))
        rel_depend_path = rel_depend_path.view(rel_depend_path.size(0) * rel_depend_path.size(1),
                                               rel_depend_path.size(2))

        path_len, path_len_sort_idx = torch.sort(path_len, descending=True)
        _, path_len_unsort_idx = torch.sort(path_len_sort_idx)
        rel_path_len, rel_path_len_sort_idx = torch.sort(rel_path_len, descending=True)
        _, rel_path_len_unsort_idx = torch.sort(rel_path_len_sort_idx)

        token_path = token_path.index_select(0, path_len_sort_idx)
        pos_path = pos_path.index_select(0, path_len_sort_idx)
        depend_path = depend_path.index_select(0, path_len_sort_idx)
        rel_token_path = rel_token_path.index_select(0, rel_path_len_sort_idx)
        rel_pos_path = rel_pos_path.index_select(0, rel_path_len_sort_idx)
        rel_depend_path = rel_depend_path.index_select(0, rel_path_len_sort_idx)

        # cut out the items with length 0
        former_len = path_len.size(0)
        num_all_paths = int(sum(sent_len))
        path_len = path_len.narrow(0, 0, num_all_paths)
        rel_path_len = rel_path_len.narrow(0, 0, num_all_paths)
        token_path = token_path.narrow(0, 0, num_all_paths)
        pos_path = pos_path.narrow(0, 0, num_all_paths)
        depend_path = depend_path.narrow(0, 0, num_all_paths)
        rel_token_path = rel_token_path.narrow(0, 0, num_all_paths)
        rel_pos_path = rel_pos_path.narrow(0, 0, num_all_paths)
        rel_depend_path = rel_depend_path.narrow(0, 0, num_all_paths)

        # Syntactic Path Representation
        token_path_embedding = self.word_repr(token_path, pos_path)
        depend_path_embedding = self.depend_embedding(depend_path)
        rel_token_path_embedding = self.word_repr(rel_token_path, rel_pos_path)
        rel_depend_path_embedding = self.depend_embedding(rel_depend_path)

        packed_token_path_embedding = pack_padded_sequence(token_path_embedding, path_len.data.numpy(),
                                                           batch_first=True)
        packed_depend_path_embedding = pack_padded_sequence(depend_path_embedding, path_len.data.numpy(),
                                                            batch_first=True)
        packed_rel_token_path_embedding = pack_padded_sequence(rel_token_path_embedding, rel_path_len.data.numpy(),
                                                               batch_first=True)
        packed_rel_depend_path_embedding = pack_padded_sequence(rel_depend_path_embedding, rel_path_len.data.numpy(),
                                                                batch_first=True)

        hidden = self.init_hidden(num_all_paths, self.config['gpr_num_layers'], self.config['gpr_hidden_size'])
        packed_token_path_repr, hidden = self.generic_path_repr_forward(packed_token_path_embedding, hidden)
        hidden = self.init_hidden(num_all_paths, self.config['rpr_num_layers'], self.config['rpr_hidden_size'])
        packed_depend_path_repr, hidden = self.relation_path_repr_forward(packed_depend_path_embedding, hidden)
        hidden = self.init_hidden(num_all_paths, self.config['gpr_num_layers'], self.config['gpr_hidden_size'])
        packed_rel_token_path_repr, hidden = self.generic_path_repr_backward(packed_rel_token_path_embedding, hidden)
        hidden = self.init_hidden(num_all_paths, self.config['rpr_num_layers'], self.config['rpr_hidden_size'])
        packed_rel_depend_path_repr, hidden = self.relation_path_repr_backward(packed_rel_depend_path_embedding, hidden)

        token_path_repr, _ = pad_packed_sequence(packed_token_path_repr, batch_first=True)
        depend_path_repr, _ = pad_packed_sequence(packed_depend_path_repr, batch_first=True)
        rel_token_path_repr, _ = pad_packed_sequence(packed_rel_token_path_repr, batch_first=True)
        rel_depend_path_repr, _ = pad_packed_sequence(packed_rel_depend_path_repr, batch_first=True)

        # select last hidden layer as feature
        token_path_repr = token_path_repr[range(0, token_path_repr.size(0)), path_len - 1, :]
        depend_path_repr = depend_path_repr[range(0, depend_path_repr.size(0)), path_len - 1, :]
        rel_token_path_repr = rel_token_path_repr[range(0, rel_token_path_repr.size(0)), path_len - 1, :]
        rel_depend_path_repr = rel_depend_path_repr[range(0, rel_depend_path_repr.size(0)), path_len - 1, :]

        gpr = torch.cat([token_path_repr, depend_path_repr], dim=-1)
        rpr = torch.cat([rel_token_path_repr, rel_depend_path_repr], dim=-1)
        spr = torch.cat([gpr, rpr], dim=-1)

        # restore the order and shape of spr
        restored_spr = Variable(torch.zeros(former_len, spr.size(1)))
        restored_spr[:spr.size(0), :] = spr
        restored_spr = restored_spr.index_select(0, path_len_unsort_idx)
        restored_spr = restored_spr.view(len(sent_len), -1, spr.size(1))
        restored_spr = restored_spr.index_select(0, sent_len_unsort_idx)

        # restore the order of gcr and concatenate gcr_e2
        gcr = gcr.index_select(0, sent_len_unsort_idx)
        gcr_e2 = gcr[range(0, gcr.size(0)), rel_pos, :].unsqueeze(1).expand_as(gcr)
        gcr = torch.cat([gcr, gcr_e2], dim=-1)

        restored_gcr = Variable(torch.zeros(former_len, gcr.size(2)))
        restored_gcr = restored_gcr.view(len(sent_len), -1, gcr.size(2))
        restored_gcr[:, :gcr.size(1), :] = gcr

        p = F.relu(self.context_repr_weight(restored_gcr)+self.path_repr_weight(restored_spr))
        pcp = self.output(p)

        return pcp

    def init_hidden(self, batch_size, num_layers, hidden_size):
        weight = next(self.parameters()).data
        return (Variable(weight.new(num_layers, batch_size, hidden_size).zero_()),
                Variable(weight.new(num_layers, batch_size, hidden_size).zero_()))

    def init_weights(self, init_range=0.1):
        self.untrained_word_embedding.weight.data.uniform_(-init_range, init_range)
        self.pos_embedding.weight.data.uniform_(-init_range, init_range)
        self.depend_embedding.weight.data.uniform_(-init_range, init_range)

    def word_repr(self, token_seq, pos_seq):
        pretrained = self.pretrained_word_embedding(token_seq)
        untrained = self.untrained_word_embedding(token_seq)
        pos_repr = self.pos_embedding(pos_seq)

        word_lex_repr = torch.cat((pretrained, untrained), dim=-1)
        word_lex_repr = torch.cat((word_lex_repr, pos_repr), dim=-1)
        word_lex_repr = F.relu(self.word_lex_repr_weight(word_lex_repr))

        return word_lex_repr


if __name__ == '__main__':
    dataset = SRLDataSet('SRL_data/data/cpbdev.txt', 'SRL_data/data/word_dict.json', 'SRL_data/data/pos_dict.json',
                         'SRL_data/data/label_dict.json', 'SRL_data/data/depend_dict.json',
                         'SRL_data/data/cpbdev_tree.txt', is_test=False)
    dataloader = DataLoader(dataset=dataset, batch_size=10)
    config = {
        'vocab_size': max(dataloader.dataset.word2idx.values()) + 1,
        'embedding_dim': 5,
        'pos_set_size': max(dataloader.dataset.pos2idx.values()) + 1,
        'depend_set_size': max(dataloader.dataset.depend2idx.values()) + 1,
        'gcr_hidden_size': 20,
        'gcr_num_layers': 1,
        'gpr_hidden_size': 20,
        'gpr_num_layers': 1,
        'rpr_hidden_size': 20,
        'rpr_num_layers': 1,
        'feature_size': 6,
        'drop_out': 0.1,
        'categories': len(dataloader.dataset.label2idx)
    }
    uf = UnifiedFramework(config)
    for d in dataloader:
        output_seq = d['output_seq']
        del (d['output_seq'])
        for k in d:
            d[k] = Variable(d[k])
        print(uf(**d))
        break
