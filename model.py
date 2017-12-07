import torch
import torch.nn as nn


class UnifiedFramework(nn.Module):
    def __init__(self, config):
        super(UnifiedFramework, self).__init__()
        self.pretrained_word_embedding = nn.Embedding(config['vocab_size'], config['embedding_dim'])
        self.untrained_word_embedding = nn.Embedding(config['vocab_size'], config['embedding_dim'])
        self.pos_embedding = nn.Embedding(config['pos_set_size'], config['embedding_dim'])
        self.syntactic_embedding = nn.Embedding(config['syn_set_size'], config['embedding_dim'])
        self.global_context_repr = nn.LSTM(config['embedding_dim'] * 3, config['gcr_hidden_size'],
                                           config['gcr_num_layers'], bidirectional=True, batch_first=True,
                                           dropout=config['drop_out'])
        self.generic_path_repr = nn.LSTM(config['embedding_dim'] * 3, config['gpr_hidden_size'],
                                         config['gpr_num_layers'], bidirectional=True, batch_first=True,
                                         dropout=config['drop_out'])
        self.relation_path_repr = nn.LSTM(config['embedding_dim'] * 3, config['rpr_hidden_size'],
                                          config['rpr_num_layers'], bidirectional=True, batch_first=True,
                                          dropout=config['drop_out'])
        self.context_repr_weight = nn.Linear(config['gcr_hidden_size'] * 4, config['feature_size'])
        self.path_repr_weight = nn.Linear(config['gpr_hidden_size'] * 2 + config['rpr_hidden_size'] * 2,
                                          config['feature_size'])
        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Softmax()
        )

    def forward(self, *input):
        pass
