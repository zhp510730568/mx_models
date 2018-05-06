import os

from collections import Iterable

PAD = '<pad>'
BOS = '<bos>'
EOS = '<eos>'
UNKNOW = '<unknow>'
VOCAB_PATH = './vocab/vocab.txt'


class Dictionary():

    def __init__(self, path, dropout=0.1):
        self.dict = {}
        self.__word_2_idx = {PAD: 0, BOS: 1, EOS: 2, UNKNOW: 3}
        self.__idx_2_word = {0: PAD, 1: BOS, 2: EOS, 3: UNKNOW}
        if os.path.exists(VOCAB_PATH):
            with open(VOCAB_PATH, 'r', buffering=10) as f:
                for line in f:
                    pair = line.strip().split(maxsplit=2)
                    if len(pair) == 2:
                        self.dict[pair[0]] = pair[1]
        else:
            with open(path, 'r', buffering=1000) as f:
                for line in f:
                    for word in line.strip():
                        self.__add_word(word)
            with open(VOCAB_PATH, 'w') as f:
                for key, value in self.dict.items():
                    f.write('%s\t%s\n' % (key, value))
        self.__vocab_size = count = int(len(self.dict) * (1.0 - dropout))
        sorted_items = sorted(self.dict.items(), key=lambda item: item[1], reverse=True)
        for idx, word in enumerate(sorted_items[0: count]):
            tmp = idx + 4
            self.__word_2_idx[word[0]] = tmp
            self.__idx_2_word[tmp] = word[0]

    def __add_word(self, word):
        if word not in self.dict.keys():
            self.dict[word] = 1
        else:
            self.dict[word] += 1

    @property
    def vocab_size(self):
        return self.__vocab_size

    @vocab_size.getter
    def vocab_size(self):
        return self.__vocab_size

    def word_2_idx(self, words):
        idxes = []
        assert isinstance(words, Iterable), 'words must by iterable type'
        for word in words:
            if word in self.__word_2_idx:
                idxes.append(self.__word_2_idx[word])
            else:
                idxes.append(self.__word_2_idx[UNKNOW])
        idxes.append(2)

        return idxes

    def idx_2_word(self, idxes):
        words = []
        assert isinstance(idxes, Iterable), 'words must by iterable type'
        for idx in idxes:
            if idx in self.__idx_2_word:
                words.append(self.__idx_2_word[idx])
            else:
                words.append(UNKNOW)

        return words


if __name__ == '__main__':
    d = Dictionary('./data/train.zh')
    print(d.vocab_size)
    case = '一对五年没见过的姐妹一场激烈的争吵？'
    print(len(case))
    idxes = d.word_2_idx(case)
    import time
    start = time.time()
    for _ in range(0, 1000000):
        d.idx_2_word(idxes)
    delay = time.time() - start
    print('total time: %d' % (delay))
    print(delay / 100000 / 213)
