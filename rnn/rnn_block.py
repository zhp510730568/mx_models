import mxnet as mx
from mxnet import gluon


class RNNLayer(gluon.HybridBlock):
    def __init__(self, **kwargs):
        super(RNNLayer, **kwargs).__init__(**kwargs)


if __name__ == '__main__':
    layer = RNNLayer()
    print(layer)
