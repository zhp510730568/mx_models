import time
from collections import defaultdict

from dictionary import Dictionary
import corpus_dataset as cds

corpus = cds.CorpusDataSet(cds.CORPUS_PATH)

start_time = time.time()

parameters = defaultdict(int)
grams = 3
with open(cds.CORPUS_PATH) as f:
    for chunk in iter(lambda: f.readline().strip(), ""):
        idxes = corpus.word_dict.word_2_idx(chunk)
        sentence_len = len(idxes)
        if sentence_len > 2:
            start = idxes[0]
            parameters[start] += 1

            for context in (idxes[i: i + (grams - 1)] for i in range(0, len(idxes) - 1)):
                t = tuple(context)
                parameters[t] += 1

            for context in (idxes[i: i + grams] for i in range(0, len(idxes) - 2)):
                t = tuple(context)
                parameters[t] += 1
        elif sentence_len == 2:
            start = idxes[0]
            parameters[start] += 1
    f.seek(0)
    count = 10
    for chunk in iter(lambda: f.readline().strip(), ""):
        idxes = corpus.word_dict.word_2_idx(chunk)
        print(idxes)
        tmp = 1.0
        for i in range(0, len(idxes) - 2):
            prop = parameters[tuple(idxes[i: i+3])] / parameters[tuple(idxes[i: i+2])]
            tmp *= prop
        if tmp > 0:
            print(pow(1 / tmp, 1 / len(idxes)))
        else:
            print('prop is zero')

    print(time.time() - start_time)