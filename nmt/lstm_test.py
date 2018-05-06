import mxnet as mx

layer = mx.gluon.rnn.LSTM(100, 10)
layer.initialize()

input = mx.nd.random.uniform(shape=(50, 3, 10))
# by default zeros are used as begin state
output = layer(input)
# manually specify begin state.
h0 = mx.nd.random.uniform(shape=(10, 3, 100))
c0 = mx.nd.random.uniform(shape=(10, 3, 100))
output, hn = layer(input, [h0, c0])

for o in output:
    print(o.shape)
for h in hn:
    print(h)