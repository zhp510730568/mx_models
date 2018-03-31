from __future__ import print_function
import mxnet as mx
import numpy as np
from mxnet import nd, autograd, gluon
mx.random.seed(1)
ctx = mx.cpu()

def dropout(X, drop_probability):
    keep_probability = 1 - drop_probability
    mask = nd.random_uniform(0, 1.0, X.shape, ctx=X.context) < keep_probability
    print(mask)
    #############################
    #  Avoid division by 0 when scaling
    #############################
    if keep_probability > 0.0:
        scale = (1/keep_probability)
    else:
        scale = 0.0
    return mask * X * scale


X = mx.nd.normal(0, 1, shape=(10, 10))


def softmax(y_linear):
    exp = nd.exp(y_linear-nd.max(y_linear))
    print(exp)
    partition = nd.nansum(exp, axis=0, exclude=True).reshape((-1,1))
    return exp / partition
print(softmax(X))
