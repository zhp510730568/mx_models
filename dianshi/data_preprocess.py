import os
import shutil
import datetime

import mxnet as mx
from mxnet import autograd
from mxnet import gluon
from mxnet import init
from mxnet import nd
from mxnet.gluon.data import vision
from mxnet.gluon import nn

import utils

data_dir = './'
train_dir = 'train_ds'
# test_dir = 'test_ds'
valid_dir = 'valid_ds'

batch_size = 64


def transform_train(data, label):
    im = data.astype('float32') / 255 - 0.5
    # auglist = image.CreateAugmenter(data_shape=(3, 112, 112), resize=0,
    #                                 rand_crop=False, rand_resize=False, rand_mirror=True,
    #                                 brightness=0, contrast=0,
    #                                 saturation=0, hue=0,
    #                                 pca_noise=0, rand_gray=0, inter_method=2)
    # for aug in auglist:
    #     im = aug(im)
    # 将数据格式从"高*宽*通道"改为"通道*高*宽"。
    im = nd.transpose(im, (2, 0, 1))
    return im, nd.array([label]).asscalar().astype('float32')


def transform_test(data, label):
    im = data.astype('float32') / 255 - 0.5
    # auglist = image.CreateAugmenter(data_shape=(3, 112, 112), resize=0,
    #                                 rand_crop=False, rand_resize=False, rand_mirror=True,
    #                                 brightness=0, contrast=0,
    #                                 saturation=0, hue=0,
    #                                 pca_noise=0, rand_gray=0, inter_method=2)
    # for aug in auglist:
    #     im = aug(im)
    im = nd.transpose(im, (2, 0, 1))
    return im, nd.array([label]).asscalar().astype('float32')


# 读取原始图像文件。flag=1说明输入图像有三个通道（彩色）。
train_ds = vision.ImageFolderDataset(data_dir + train_dir, flag=1,
                                     transform=transform_train)

valid_ds = vision.ImageFolderDataset(data_dir + valid_dir, flag=1,
                                     transform=transform_test)

loader = gluon.data.DataLoader
train_data = loader(train_ds, batch_size, shuffle=True, last_batch='keep', num_workers=6)
valid_data = loader(valid_ds, batch_size, shuffle=True, last_batch='keep', num_workers=6)

# 交叉熵损失函数。
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()


class Residual(nn.HybridBlock):
    def __init__(self, channels, same_shape=True, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.same_shape = same_shape
        with self.name_scope():
            strides = 1 if same_shape else 2
            self.conv1 = nn.Conv2D(channels, kernel_size=3, padding=1,
                                   strides=strides)
            self.bn1 = nn.BatchNorm()
            self.conv2 = nn.Conv2D(channels, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm()
            if not same_shape:
                self.conv3 = nn.Conv2D(channels, kernel_size=1,
                                       strides=strides)

    def hybrid_forward(self, F, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if not self.same_shape:
            x = self.conv3(x)
        return F.relu(out + x)


class ResNet(nn.HybridBlock):
    def __init__(self, num_classes, verbose=False, **kwargs):
        super(ResNet, self).__init__(**kwargs)
        self.verbose = verbose
        with self.name_scope():
            net = self.net = nn.HybridSequential()
            net.add(
                # 第一阶段
                nn.Conv2D(channels=24, kernel_size=3,
                          strides=4, activation='relu'),
                nn.MaxPool2D(pool_size=3, strides=2),
                # 第二阶段
                nn.Conv2D(channels=48, kernel_size=3,
                          padding=2, activation='relu'),
                nn.MaxPool2D(pool_size=3, strides=2),
                # 第三阶段
                nn.Conv2D(channels=48, kernel_size=3,
                          padding=2, activation='relu'),
                nn.Conv2D(channels=48, kernel_size=3,
                          padding=2, activation='relu'),
                nn.MaxPool2D(pool_size=3, strides=2),
                # 第四阶段
                nn.Flatten(),
                nn.Dense(1024, activation="relu"),
                # 第五阶段
                nn.Dense(1024, activation="relu"),
                # 第五阶段
                nn.Dense(1024, activation="relu"),
                # 第六阶段
                nn.Dense(num_classes)
            )

    def hybrid_forward(self, F, x):
        out = x
        for i, b in enumerate(self.net):
            print(b)
            out = b(out)
            if self.verbose:
                print('Block %d output: %s' % (i + 1, out.shape))
        return out


def get_net(ctx):
    num_outputs = 100
    net = ResNet(num_outputs)
    net.initialize(ctx=ctx, init=init.Xavier())
    return net


def train(net, train_data, valid_data, num_epochs, lr, wd, ctx, lr_period, lr_decay):
    trainer = gluon.Trainer(
        net.collect_params(), 'sgd', {'learning_rate': lr, 'momentum': 0.99, 'wd': wd})

    prev_time = datetime.datetime.now()
    for epoch in range(num_epochs):
        print(epoch)
        train_loss = 0.0
        train_acc = 0.0
        if epoch > 0 and epoch % lr_period == 0:
            trainer.set_learning_rate(trainer.learning_rate * lr_decay)
        count = 0
        for data, label in train_data:
            label = label.as_in_context(ctx)
            with autograd.record():
                output = net(data.as_in_context(ctx))
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(batch_size)
            count += 1
            train_loss += nd.mean(loss).asscalar()
            train_acc += utils.accuracy(output, label)
        cur_time = datetime.datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        if valid_data is not None:
            valid_acc = utils.evaluate_accuracy(valid_data, net, ctx)
            epoch_str = ("Epoch %d. Loss: %f, Train acc %f, Valid acc %f, "
                         % (epoch, train_loss / len(train_data),
                            train_acc / len(train_data), valid_acc))
        else:
            epoch_str = ("Epoch %d. Loss: %f, Train acc %f, "
                         % (epoch, train_loss / len(train_data),
                            train_acc / len(train_data)))
        prev_time = cur_time
        print(epoch_str + time_str + ', lr ' + str(trainer.learning_rate))


ctx = mx.gpu()
num_epochs = 10000
learning_rate = 0.01
weight_decay = 5e-4
lr_period = 50
lr_decay = 0.1

net = get_net(ctx)
net.hybridize()
train(net, train_data, valid_data, num_epochs, learning_rate, weight_decay, ctx, lr_period, lr_decay)