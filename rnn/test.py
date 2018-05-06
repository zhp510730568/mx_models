import numpy as np
import mxnet as mx

from mxnet import gluon
from mxnet import initializer

h2h_weight_initializer = initializer.Constant(value=[[0.1, 0.2], [0.3, 0.4]])
h2h_bias_initializer = initializer.Constant(value=[0.1, -0.1])

i2h_weight_initializer = initializer.Constant(value=[[0.5], [0.6]])
# i2h_bias_initializer = initializer.Constant(value=[0.1, -0.1])

rnn = gluon.rnn.RNN(hidden_size=2, num_layers=1, h2h_weight_initializer=h2h_weight_initializer,
                    h2h_bias_initializer=h2h_bias_initializer, )

rnn.initialize()

input = mx.nd.array([[1.0, 2.0]], dtype=np.float32)
input = mx.nd.swapaxes(input, dim1=1, dim2=0)
print('input: ', input)
h0 = mx.nd.array([[[0.0, 0.0]]])
c = mx.nd.array([[[0.0, 0.0]]])
output = rnn(input, [h0, c])
print('state: ', output)

hidden = mx.nd.array([0.53704959, 0.46211717, 2.0])
weight = mx.nd.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

output = mx.nd.dot(hidden, weight) + mx.nd.array([0.1, -0.1])
print(mx.nd.tanh(output))