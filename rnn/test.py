import mxnet as mx
ctx = mx.gpu(0)
x = mx.nd.zeros((16, 100), ctx=ctx)
w = mx.gluon.Parameter('fc_weight', shape=(64, 100), init=mx.init.Xavier())
b = mx.gluon.Parameter('fc_bias', shape=(64,), init=mx.init.Constant(1.2))
w.initialize(ctx=ctx)
b.initialize(ctx=ctx)
out = mx.nd.FullyConnected(x, w.data(ctx), b.data(ctx), num_hidden=64)
print(w.data())
print(out)