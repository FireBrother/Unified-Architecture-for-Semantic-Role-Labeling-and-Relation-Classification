import IPython
import time

import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F

from data_helper import SRLDataSet

testset = SRLDataSet('SRL_data/data/cpbtest.txt', 'SRL_data/data/word_dict.json', 'SRL_data/data/pos_dict.json',
                      'SRL_data/data/label_dict.json', 'SRL_data/data/depend_dict.json',
                      'SRL_data/data/cpbtest_tree.txt', is_test=True)
testloader = DataLoader(dataset=testset, batch_size=16)
devset = SRLDataSet('SRL_data/data/cpbdev.txt', 'SRL_data/data/word_dict.json', 'SRL_data/data/pos_dict.json',
                    'SRL_data/data/label_dict.json', 'SRL_data/data/depend_dict.json',
                    'SRL_data/data/cpbdev_tree.txt', is_test=False)
devloader = DataLoader(dataset=devset, batch_size=4)

idx2label = {v: k for k, v in testloader.dataset.label2idx.items()}
idx2word = {v: k for k, v in testloader.dataset.word2idx.items()}
idx2pos = {v: k for k, v in testloader.dataset.pos2idx.items()}

PROB_THRESH = 0.3

with open('model.pkl', 'rb') as f:
    uf = torch.load(f)

DEVICE_NO = 0
if DEVICE_NO != -1:
    uf = uf.cuda(DEVICE_NO)


def sample(dataloader):
    for batch in dataloader:
        output_seq = Variable(batch['output_seq'])
        del (batch['output_seq'])
        for k in batch:
            batch[k] = Variable(batch[k])
        if DEVICE_NO != -1:
            output_seq = output_seq.cuda(DEVICE_NO)
            for k in batch:
                batch[k] = batch[k].cuda(DEVICE_NO)
        pred = uf.forward(**batch)
        pred = F.softmax(pred, dim=-1)
        prob, label = torch.max(pred, dim=-1)
        for i in range(len(list(batch.values())[0])):
            for j in range(int(batch['sent_len'][i])):
                word = idx2word[int(batch['word_seq'][i, int(j)])]
                l_true = idx2label[int(output_seq[i, int(j)])]
                p = float(prob[i, int(j)])
                l = idx2label[int(label[i, int(j)])] if p > PROB_THRESH else 'O'
                print('{}/{}/{}'.format(word, l_true, l), end=' ')
            print('')
        input('input to continue:')


def test(dataloader, out=sys.stdout):
    for batch in dataloader:
        if 'output_seq' in batch:
            del batch['output_seq']
        for k in batch:
            batch[k] = Variable(batch[k])
        if DEVICE_NO != -1:
            for k in batch:
                batch[k] = batch[k].cuda(DEVICE_NO)
        pred = uf.forward(**batch)
        pred = F.softmax(pred, dim=-1)
        prob, label = torch.max(pred, dim=-1)
        for i in range(len(list(batch.values())[0])):
            for j in range(int(batch['sent_len'][i])):
                word = idx2word[int(batch['word_seq'][i, int(j)])]
                pos = idx2pos[int(batch['pos_seq'][i, int(j)])]
                p = float(prob[i, int(j)])
                l = idx2label[int(label[i, int(j)])] if p > PROB_THRESH else 'O'
                out.write('{}/{}/{} '.format(word, pos, l))
            out.write('\n')


sample(devloader)
# test(testloader, out=open('test_output.txt', 'w'))
