import numpy as np
import mxnet as mx

h2h_weight = mx.nd.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
h2h_bias = mx.nd.array([0.1, -0.1], dtype=np.float32)

i2h_weight = mx.nd.array([[0.5, 0.6]])

X = mx.nd.array([[1.0], [2.0]])

state = mx.nd.array([0.0, 0.0])

h2o_weight = mx.nd.array([1.0, 2.0], dtype=np.float32)
h2o_bias = mx.nd.array([0.1])

for i in range(len(X)):
    brfore_activation = mx.nd.dot(state, h2h_weight) + X[i] * i2h_weight + h2h_bias
    state = mx.nd.tanh(brfore_activation)
    output = mx.nd.dot(state, h2o_weight) + h2o_bias
    print(output)
