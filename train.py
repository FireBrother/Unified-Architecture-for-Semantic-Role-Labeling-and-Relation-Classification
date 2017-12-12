import IPython
import time

import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader

from data_helper import SRLDataSet
from model import UnifiedFramework

trainset = SRLDataSet('SRL_data/data/cpbtrain.txt', 'SRL_data/data/word_dict.json', 'SRL_data/data/pos_dict.json',
                      'SRL_data/data/label_dict.json', 'SRL_data/data/depend_dict.json',
                      'SRL_data/data/cpbtrain_tree.txt', is_test=False)
trainloader = DataLoader(dataset=trainset, batch_size=16, shuffle=True)
devset = SRLDataSet('SRL_data/data/cpbdev.txt', 'SRL_data/data/word_dict.json', 'SRL_data/data/pos_dict.json',
                    'SRL_data/data/label_dict.json', 'SRL_data/data/depend_dict.json',
                    'SRL_data/data/cpbdev_tree.txt', is_test=False)
devloader = DataLoader(dataset=devset, batch_size=16)
config = {
    'vocab_size': max(trainloader.dataset.word2idx.values()) + 1,
    'embedding_dim': 5,
    'pos_set_size': max(trainloader.dataset.pos2idx.values()) + 1,
    'depend_set_size': max(trainloader.dataset.depend2idx.values()) + 1,
    'gcr_hidden_size': 20,
    'gcr_num_layers': 1,
    'gpr_hidden_size': 20,
    'gpr_num_layers': 1,
    'rpr_hidden_size': 20,
    'rpr_num_layers': 1,
    'feature_size': 6,
    'drop_out': 0.1,
    'categories': max(trainloader.dataset.label2idx.values()) + 1
}

uf = UnifiedFramework(config)

DEVICE_NO = 1
if DEVICE_NO != -1:
    uf = uf.cuda(DEVICE_NO)

optimizer = torch.optim.Adagrad(uf.parameters(), lr=0.01)
criteria = nn.CrossEntropyLoss(ignore_index=0)

log_interval = 10
epochs = 10


def train(dataloader):
    uf.train()
    total_loss = 0
    total_items = 0
    uf.init_weights()
    start_time = time.time()
    for i_batch, batch in enumerate(dataloader):
        output_seq = Variable(batch['output_seq'])
        del (batch['output_seq'])
        for k in batch:
            batch[k] = Variable(batch[k])
        if DEVICE_NO != -1:
            output_seq = output_seq.cuda(DEVICE_NO)
            for k in batch:
                batch[k] = batch[k].cuda(DEVICE_NO)
        uf.zero_grad()
        pred = uf.forward(**batch)
        pred = pred.view(-1, pred.size(-1))
        output_seq = output_seq.view(-1)
        loss = criteria(pred, output_seq)
        loss.backward()
        num_items = len([x for x in output_seq if int(x) != criteria.ignore_index])
        total_loss += num_items * loss.data
        total_items += num_items
        optimizer.step()

        if i_batch % log_interval == 0 and i_batch > 0:
            cur_loss = total_loss[0] / total_items
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:04.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, i_batch, len(dataloader.dataset) // dataloader.batch_size, optimizer.param_groups[0]['lr'],
                                elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            total_items = 0
            start_time = time.time()


def evaluate(dataloader):
    total_loss = 0
    total_items = 0
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
        pred = pred.view(-1, pred.size(-1))
        output_seq = output_seq.view(-1)
        num_items = len([x for x in output_seq if int(x) != criteria.ignore_index])
        total_loss += num_items * criteria(pred, output_seq).data
        total_items += num_items

    return total_loss[0] / total_items


best_val_loss = 1000
try:
    for epoch in range(1, epochs + 1):
        # scheduler.step()
        epoch_start_time = time.time()
        train(trainloader)
        val_loss = evaluate(devloader)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            print('new best val loss, saving model')
            with open('model.pkl', 'wb') as f:
                torch.save(uf, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            pass
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')
    val_loss = evaluate(devloader)
    # Save the model if the validation loss is the best we've seen so far.
    if not best_val_loss or val_loss < best_val_loss:
        print('new best val loss, saving model')
        with open('model.pkl', 'wb') as f:
            torch.save(uf, f)
        best_val_loss = val_loss
    else:
        # Anneal the learning rate if no improvement has been seen in the validation dataset.
        pass
