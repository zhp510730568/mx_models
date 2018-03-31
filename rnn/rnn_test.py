import mxnet as mx
import mxnet.autograd as ag

ctx = mx.cpu()

input = mx.nd.ones(shape=(10, 50))

xh = mx.nd.random_normal(0, 0.01, shape=(50, 100), ctx=ctx)
hh = mx.nd.random_normal(0, 0.01, shape=(100, 100), ctx=ctx)
hy = mx.nd.random_normal(0, 0.01, shape=(100, 10), ctx=ctx)

params = [xh, hh, hy]

for param in params:
    param.attach_grad()

h = mx.nd.dot(input, xh)
outputs = []
for idx in range(input.shape[0]):
    output = mx.nd.relu(mx.nd.dot(input[idx, :].reshape([1, 50]), xh))
    print(output)
output = mx.nd.dot(h, hh)
yhat = mx.nd.dot(output, hy)
print(yhat)