import json

import IPython
import gensim
import itertools
from gensim.models.word2vec import LineSentence
from gensim.models import Word2Vec
import numpy as np
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

corpus_filename = 'xinhuashe/xinhuashe.txt'
MAX_WORDS_IN_BATCH = 10000


class MyLineSentence(object):
    def __init__(self, source, max_sentence_length=MAX_WORDS_IN_BATCH, limit=None):
        self.source = source
        self.max_sentence_length = max_sentence_length
        self.limit = limit

    def __iter__(self):
        try:
            # Assume it is a file-like object and try treating it as such
            # Things that don't have seek will trigger an exception
            self.source.seek(0)
            for line in itertools.islice(self.source, self.limit):
                line = gensim.utils.to_unicode(line).split()
                i = 0
                while i < len(line):
                    yield [x.split('/')[0] for x in line[i: i + self.max_sentence_length]]
                    i += self.max_sentence_length
        except AttributeError:
            # If it didn't work like a file, use it as a string filename
            with gensim.utils.smart_open(self.source) as fin:
                for line in itertools.islice(fin, self.limit):
                    line = gensim.utils.to_unicode(line).split()
                    i = 0
                    while i < len(line):
                        yield [x.split('/')[0] for x in line[i: i + self.max_sentence_length]]
                        i += self.max_sentence_length


# Train word2vec
# sentences = MyLineSentence(corpus_filename)
# model = Word2Vec(sentences, size=100, workers=12)
# model.save('xinhuashe/D100.wv')

model = Word2Vec.load('xinhuashe/D100.wv')
_word_set = sorted(json.load(open('SRL_data/data/word_dict.json')).keys())
idx2word = {i + 1: w for i, w in enumerate(_word_set)}

weights = np.zeros((max(idx2word.keys()) + 1, 100))
oov_num = 0
for i in range(1, max(idx2word.keys()) + 1):
    if idx2word[i] in model.wv:
        weights[i, :] = model.wv[idx2word[i]]
    else:
        oov_num += 1
np.save('weights.pkl', weights)
print('total:{} oov:{}'.format(weights.shape[0], oov_num))
