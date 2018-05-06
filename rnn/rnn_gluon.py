import mxnet as mx

model = mx.gluon.nn.Sequential()
with model.name_scope():
    model.add(mx.gluon.nn.Embedding(30, 10))
    lstm = mx.gluon.rnn.LSTM(20)

    model.add(lstm)
    model.add(mx.gluon.nn.Dense(5, flatten=False))
model.initialize()

net = model(mx.nd.ones((2,3)))
print(net)
loss = mx.gluon.loss.ndarray.sum(net, axis=0)

trainer = mx.gluon.Trainer(
    model.collect_params(), 'sgd', {'learning_rate': 0.01, 'momentum': 0.9, 'wd': 0.99})
print(type(model.collect_params()['sequential0_lstm0_l0_h2h_weight']))