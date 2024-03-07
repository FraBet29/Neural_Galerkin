import torch

# TEST 1

a = torch.tensor([2., 3.], requires_grad=True) # points where we want to compute the gradient
b = torch.tensor([6., 4.], requires_grad=True) # points where we want to compute the gradient
Q = 3 * a ** 3 - b ** 2
# print(a.grad)
# print(b.grad)
# print(Q)

# We need to explicitly pass a gradient argument in Q.backward() because it is a vector
# Gradient is a tensor of the same shape as Q, and it represents the gradient of Q w.r.t. itself
# external_grad = torch.tensor([1., 1.])
# Q.backward(gradient=external_grad)

# Equivalently, we can also aggregate Q into a scalar and call backward implicitly
Q.sum().backward()

# print(Q)
# print(a.grad)
# print(b.grad)

# Check if collected gradients are correct
# print(9 * a ** 2 == a.grad)
# print(- 2 * b == b.grad)

# Test with a simple neural network
# y = w * x + b
# dy / dw_i = x_i, dy / db_i = 1
# x.shape = (batch, input), w.shape = (output, input), b.shape = (output)

w = torch.tensor([[1., 2.]], requires_grad=True)
b = torch.tensor([[3.]], requires_grad=True)
x = torch.tensor([[4., 5.]], requires_grad=True)

model = torch.nn.Linear(2, 1)
model.weight = torch.nn.Parameter(w)
model.bias = torch.nn.Parameter(b)

y = model(x)
print(y)

y.sum().backward()
print("dw:", model.weight.grad)
print("db:", model.bias.grad)
# y.sum().backward()

# TEST 2

def dw(f): # must be evaluated on a tensor!
	return torch.func.grad(f)

def db(f): # must be evaluated on a tensor!
	return torch.func.grad(f, argnums=1)

def ddw(f): # must be evaluated on a tensor!
	return torch.func.grad(torch.func.grad(f))

w = torch.tensor([[1., 2.]], requires_grad=True)
b = torch.tensor([[3.]], requires_grad=True)
x = torch.tensor([[4., 5.]], requires_grad=True)

def compute_y(w, b, x):
	model = torch.nn.Linear(2, 1)
	model.weight = torch.nn.Parameter(w)
	model.bias = torch.nn.Parameter(b)
	y = model(x)
	print(y.shape)
	return y.squeeze()

y = compute_y(w, b, x)
print(y)

dy_dw = dw(compute_y)(w, b, x)
dy_db = db(compute_y)(w, b, x)
print("dy_dw:", dy_dw)
print("dy_db:", dy_db)

# TEST 3

w = torch.tensor([[1., 2.]], requires_grad=True)
b = torch.tensor([[3.]], requires_grad=True)
x = torch.tensor([[4., 5.]], requires_grad=True)

model = torch.nn.Linear(2, 1)
model.weight = torch.nn.Parameter(w)
model.bias = torch.nn.Parameter(b)

def f(w, b, x):
	y = w @ x.T + b
	return y

# y = model(x).reshape(-1)
# print(y)

# dw_auto = torch.autograd.grad(y, w, create_graph=True, allow_unused=True)[0]
# print("dw_auto:", dw_auto)

# ddw_auto = torch.autograd.grad(dw_auto, w, allow_unused=True)[0]
# print("ddw_auto:", ddw_auto)

y = f(w, b, x)
print(y)

dw_auto = torch.autograd.grad(y, w, create_graph=True, allow_unused=True)[0]
print("dw_auto:", dw_auto)

dw_auto_first = dw_auto[0][0]
print("dw_auto_first:", dw_auto_first)

ddw_auto = torch.autograd.grad(dw_auto_first, w, allow_unused=True)[0]
print("ddw_auto:", ddw_auto)