import mxnet as mx
from mxnet import gluon
from mxnet import autograd

net = gluon.nn.Sequential()
net.add(gluon.nn.Dense(10))
net.add(gluon.nn.Activation('sigmoid'))
net.add(gluon.nn.Dropout(0.9))
net.add(gluon.nn.Dense(10))
net.add(gluon.nn.Activation('sigmoid'))

params = net.collect_params()
net.initialize(ctx=mx.gpu(), init=mx.init.Xavier())

print(params)

lr = 1
wd = 5e-3
ctx = mx.gpu()

data = mx.random.uniform(0, 10, shape=(10, 10))

params = net.collect_params()
trainer = gluon.Trainer(
    params, 'sgd', {'learning_rate': lr, 'momentum': 0.9, 'wd': wd})
for _ in range(10):
    with autograd.record():
        output = net(data.as_in_context(ctx))
    output.backward()
    trainer.step(10)
    print('grad: ', params['dense0_weight'].grad())
    print('data: ', params['dense0_weight'].data())
print(output)
