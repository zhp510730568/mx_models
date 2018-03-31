import mxnet as mx

mx.random.seed(1)

x = mx.nd.array([[1, 2], [3, 4]])

x.attach_grad(grad_req='write')

for _ in range(10):
    with mx.autograd.record():
        y = x * 2
        z = y * x
    head_gradient = mx.nd.array([[10, 1.], [.1, .01]])
    z.backward(head_gradient)
    print(x.grad)
