import os
import time

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize.stanford_segmenter import StanfordSegmenter

UNK = '<UNK>'
BOS = '<BOS>'
EOS = '<EOS>'

data_path = './data'
file_name = 'ai_challenger_translation_train_20170912.zip'
corpora_path = 'ai_challenger_translation_train_20170912/translation_train_20170912'
en_corpora_name = 'train.en'
zh_corpora_name = 'train.zh'

count = 0
start = time
vocab = set()
word_count = {}
lines = 0
with open(os.path.join(data_path, corpora_path, en_corpora_name)) as en:
    for idx, line in enumerate(en):
        lines += 1
        tokens = word_tokenize(line)
        for token in tokens:
            if token in word_count:
                word_count[token] = word_count[token] + 1
            else:
                word_count[token] = 1
        if lines == 10000:
            break
word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
with open(os.path.join(data_path, 'vocab.txt'), 'w') as f:
    for key, count in word_count:
        f.write('%s\t%s\n' % (key, count))

print('total count: %d' % count)