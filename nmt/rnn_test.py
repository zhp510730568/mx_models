import mxnet as mx

layer = mx.gluon.rnn.RNN(100, 5, activation='tanh')
layer.initialize()

input = mx.nd.random.uniform(shape=(50, 3, 10))
# by default zeros are used as begin state
output = layer(input)
# manually specify begin state.
h0 = mx.nd.random.uniform(shape=(5, 3, 100))
c0 = mx.nd.random.uniform(shape=(5, 3, 100))
output, hn = layer(input, [h0, c0])


for h in output:
    print(h.shape)

for h in hn:
    print(h.shape)
print(len(hn))