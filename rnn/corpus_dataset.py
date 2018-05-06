import os, sys, time
from collections import namedtuple

import mxnet.ndarray as nd

from dictionary import Dictionary

DataBatch = namedtuple('data_batch', ['data', 'label'])
CORPUS_PATH = './data/train.zh'


class CorpusDataSet():
    def __init__(self, path, step=3):
        assert os.path.exists(path), 'corpus is not exist!!'
        self.path = path
        self.step = step
        try:
            self.file = open(self.path, 'r')
        except FileNotFoundError as err:
            sys.stderr.write(err.args[1])
            sys.exit(-1)

        self.word_dict = Dictionary(path)
        self.line = ''

    def hasNext(self):
        self.line = str.strip(self.file.readline())
        return self.line != ''

    def next(self):
        return self.line

    def reset(self):
        self.file.close()
        self.file = open(self.path, 'r')


if __name__ == '__main__':
    corpus = CorpusDataSet(CORPUS_PATH)
    print(corpus.word_dict.vocab_size)
    start = time.time()
    with open(CORPUS_PATH) as f:
        for chunk in iter(lambda: f.readline().strip(), ""):
            idxes = corpus.word_dict.word_2_idx(chunk)
            print(idxes)


    print('total time %d' % (time.time() - start))