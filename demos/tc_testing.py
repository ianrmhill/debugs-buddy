import torch as tc

batch_dims = (4, 5, 3)
inputs = tc.rand(batch_dims, dtype=tc.float)
print(inputs[..., :-1])
print(inputs[..., -2:-1])

a = tc.zeros((*inputs.shape[:-1], 4), dtype=tc.float)
b = tc.ones((*inputs.shape[:-1], 4), dtype=tc.float)
print(a)
print(b)

# Stack along the 2nd last dimension, so that we have batch dims, then the stack of rows of coefficients
c = tc.stack([a, b], -2)
print(c)