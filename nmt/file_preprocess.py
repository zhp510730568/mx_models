import os
import zipfile

data_path = './data'
file_name = 'ai_challenger_translation_train_20170912.zip'
corpora_path = 'ai_challenger_translation_train_20170912/translation_train_20170912'
en_corpora_name = 'train.en'
zh_corpora_name = 'train.zh'


def extracecorpora(file_path):
    with zipfile.ZipFile(os.path.join(data_path, file_path), mode='r') as f:
        f.extractall(data_path)

