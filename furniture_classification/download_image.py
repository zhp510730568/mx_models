import os
import shutil
import threading

import json
import urllib.request

import redis

import mxnet as mx

root_path = './data'
train_file = 'train.json'

data_dir = './'
train_dir = 'valid_ds'
test_dir = 'train_ds'


def downloadimage(url, label, image_id):
    '''
    download images, and save label directory
    :param url:
    :param label:
    :return:
    '''
    filename = os.path.basename(url)
    arr = os.path.splitext(filename)
    response = urllib.request.urlopen(url)
    img = response.read()
    filename = '%s_%s%s' % (label, image_id, arr[-1].split('?')[0])
    path = os.path.join(train_dir, label, filename)
    if not os.path.exists(path):
        with open(path, 'wb') as f:
            f.write(img)


def checkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


class DownloadThread(threading.Thread):
    def __init__(self, train_annotations, train_urls, start_index, end_index):
        threading.Thread.__init__(self)
        self.train_annotations = train_annotations
        self.train_urls = train_urls
        self.start_index = start_index
        self.end_index = end_index

    def run(self):
        cli = redis.Redis(host='127.0.0.1', port=6379)
        count = 0
        while True:
            meta_json = cli.rpop('furniture_set')
            if meta_json is not None:
                meta = json.loads(meta_json)
                label = str(meta['label_id'])
                image_id = str(meta['image_id'])
                url = meta['url']
                checkdir(os.path.join(train_dir, label))
                try:
                    downloadimage(url, label, image_id)
                    count += 1
                    if count % 1000 == 0:
                        print('count: %s' % count)
                except Exception as err:
                    print(err)
                    meta['try'] += 1
                    cli.lpush('furniture_set', json.dumps(meta))


def downloadtrainset():
    with open(os.path.join(root_path, train_file)) as f:
        content = f.read()
    train_images = json.loads(content)

    train_annotations = train_images['annotations']
    train_urls = train_images['images']

    count = len(train_annotations)
    batchs = 24
    d, m = divmod(count, batchs)

    if m > 0:
        d = d + 1
    downloadthreads = []
    for batch in range(0, batchs):
        if batch * d <= (count - d):
            downloadthread = DownloadThread(train_annotations, train_urls, batch * d, (batch + 1) * d)
            downloadthreads.append(downloadthread)
        else:
            downloadthread = DownloadThread(train_annotations, train_urls, batch * d, -1)
            downloadthreads.append(downloadthread)

    for thd in downloadthreads:
        thd.start()

    for thd in downloadthreads:
        thd.join()


def reorg_funiture_data(data_dir, label_file, train_dir, test_dir, input_dir, valid_ratio):
    # 读取训练数据标签。
    with open(os.path.join(data_dir, label_file), 'r') as f:
        # 跳过文件头行（栏名称）。
        lines = f.readlines()[1:]
        tokens = [l.rstrip().split(',') for l in lines]
        idx_label = dict(((int(idx), label) for idx, label in tokens))
    labels = set(idx_label.values())

    num_train = len(os.listdir(os.path.join(data_dir, train_dir)))
    num_train_tuning = int(num_train * (1 - valid_ratio))
    assert 0 < num_train_tuning < num_train
    num_train_tuning_per_label = num_train_tuning // len(labels)
    label_count = dict()

    def mkdir_if_not_exist(path):
        if not os.path.exists(os.path.join(*path)):
            os.makedirs(os.path.join(*path))

    for train_file in os.listdir(os.path.join(data_dir, train_dir)):
        idx = int(train_file.split('.')[0])
        label = idx_label[idx]
        mkdir_if_not_exist([data_dir, input_dir, 'train_valid', label])
        shutil.copy(os.path.join(data_dir, train_dir, train_file),
                    os.path.join(data_dir, input_dir, 'train_valid', label))
        if label not in label_count or label_count[label] < num_train_tuning_per_label:
            mkdir_if_not_exist([data_dir, input_dir, 'train', label])
            shutil.copy(os.path.join(data_dir, train_dir, train_file),
                        os.path.join(data_dir, input_dir, 'train', label))
            label_count[label] = label_count.get(label, 0) + 1
        else:
            mkdir_if_not_exist([data_dir, input_dir, 'valid', label])
            shutil.copy(os.path.join(data_dir, train_dir, train_file),
                        os.path.join(data_dir, input_dir, 'valid', label))

    mkdir_if_not_exist([data_dir, input_dir, 'test', 'unknown'])
    for test_file in os.listdir(os.path.join(data_dir, test_dir)):
        shutil.copy(os.path.join(data_dir, test_dir, test_file),
                    os.path.join(data_dir, input_dir, 'test', 'unknown'))


def pushmeta():
    cli = redis.Redis(host='127.0.0.1', port=6379)
    with open(os.path.join(root_path, train_file)) as f:
        content = f.read()
    train_images = json.loads(content)
    for annotation, url in zip(train_images['annotations'], train_images['images']):
        image_meta = {'image_id': annotation['image_id'], 'label_id': annotation['label_id'], 'url': url['url'][0], 'try': 0}
        result = cli.lpush('furniture_set', json.dumps(image_meta))
        print(result)


with open(os.path.join(root_path, 'image.lst'), 'w') as f:
    index = 0
    for label_dir in os.listdir(train_dir):
        label = label_dir
        for image_file in os.listdir(os.path.join(train_dir, label_dir)):
            if image_file.endswith('.jpg') or image_file.endswith('.jpeg') or image_file.endswith('.png'):
                index += 1
                image_path = os.path.join(train_dir, label_dir, image_file)
                image_path = os.path.abspath(image_path)
                f.write('%d\t%s\t%s\n' % (index, label, image_path))