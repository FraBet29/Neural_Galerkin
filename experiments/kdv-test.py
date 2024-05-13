import numpy as np
import torch
import torch.nn as nn
# from typing import Callable
import matplotlib.pyplot as plt


####### Automatic differentiation #######

# https://pytorch.org/docs/stable/func.html

# def dx(fun: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], x, t):
#     fun_x = lambda x: fun(x, t)
#     fun_x(x).backward()
#     return x.grad

# def ddx(f: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], x, t):
# 	# h = torch.func.hessian(f, (0, 1)) # full hessian w.r.t. x and t
# 	h = torch.func.hessian(f) # second derivative w.r.t. x
# 	return h(x, t)

# def ddx(f, x, t):
	# h = torch.func.hessian(f, (0, 1)) # full hessian w.r.t. x and t
	# h = torch.func.hessian(f) # second derivative w.r.t. x
	# assert x.shape == t.shape, "x and t must have the same shape"
	# df = torch.func.grad(f) # first derivative w.r.t. x
	# ddf = torch.func.grad(df) # second derivative w.r.t. x
    # ddf = torch.func.hessian(f)
    # return ddf
	# if x.shape == () and t.shape == (): # scalar space and time
	# 	return ddf(x, t)
	# elif len(x.shape) == 1 and t.shape == (): # 1D space and scalar time
	# 	return torch.stack([ddf(x[i], t) for i in range(x.shape[0])])
	# elif x.shape == () and len(t.shape) == 1: # scalar space and 1D time
	# 	return torch.stack([ddf(x, t[i]) for i in range(t.shape[0])])
	# elif len(x.shape) == 1 and len(t.shape) == 1: # 1D space and 1D time
	# 	assert x.shape == t.shape, "x and t must have the same shape"
	# 	return torch.stack([ddf(x[i], t[i]) for i in range(x.shape[0])])
	# elif len(x.shape) == 2 and t.shape == (): # 2D space and scalar time
	# 	return torch.stack([ddf(x[i, j], t) for i in range(x.shape[0]) for j in range(x.shape[1])]).view(x.shape)
	# elif len(x.shape) == 2 and len(t.shape) == 2: # 2D space and 2D time (useful for plots)
	# 	assert x.shape == t.shape, "x and t must have the same shape"
	# 	return torch.stack([ddf(x[i, j], t[i, j]) for i in range(x.shape[0]) for j in range(x.shape[1])]).view(x.shape)
	# else:
	# 	raise ValueError("x and t must be scalars, 1D or 2D tensors")

def dx(f): # must be evaluated on a tensor!
	return torch.func.grad(f)

def ddx(f): # must be evaluated on a tensor!
	return torch.func.grad(torch.func.grad(f))


# Test automatic differentiation
# def f(x, t):
# 	return x ** 4 + t ** 3

# x = torch.tensor(2.0)
# t = torch.tensor(1.0)

# print("df_x(x, t) =", dx(f)(x, t))
# print("ddf_x(f, x, t) =", ddx(f)(x, t))

##################################################################################################################
####################################### KdV equation with two solitons ###########################################
##################################################################################################################

####### Problem data #######

# Space-time domain
X_lower = -20
X_upper = 40
T_lower = 0
T_upper = 4

# NN architecture
d = 1 # input dimension
m = 10 # nodes
L = 60 # parameter for the gaussian kernel
n = 1000 # samples
batch_size = 10 ** 5 # batch size (since n = 1000, we will take the whole dataset?)
epochs = 10 ** 5 # number of epochs
gamma = 0.1 # learning rate


####### Exact solution #######
# k1 = 1
# k2 = 5 ** (1/2)
# eta1_0 = 0
# eta2_0 = 10.73
# eta1 = lambda x, t: k1 * x - k1 ** 3 * t + eta1_0
# eta2 = lambda x, t: k2 * x - k2 ** 3 * t + eta2_0
# g = lambda x, t: 1 + torch.exp(eta1(x, t)) + torch.exp(eta2(x, t)) + torch.exp(eta1(x, t) + eta2(x, t)) * ((k1 - k2) / (k1 + k2)) ** 2
# u = lambda x, t: 2 * ddx(lambda x, t: torch.log(g(x, t)), x, t)
# u_0 = lambda x: u(x, torch.tensor(0.0))

def exactKdVTwoSol(x, t):
    '''
    Function taken from https://github.com/pehersto/ng (exactKdV.py)
    '''

    k = np.asarray([1., np.sqrt(5.)])
    eta = np.asarray([0., 10.73])
    t = np.asarray(t)

    etaMat1 = k[0] * x.reshape((-1, 1)) - k[0] ** 3 * t.reshape((1, -1)) + eta[0]
    etaMat2 = k[1] * x.reshape((-1, 1)) - k[1] ** 3 * t.reshape((1, -1)) + eta[1]
    c = ((k[0] - k[1]) / (k[0] + k[1]) )** 2

    f = 1. + np.exp(etaMat1) + np.exp(etaMat2) + np.multiply(np.exp(etaMat1), np.exp(etaMat2) * c)
    df = k[0] * np.exp(etaMat1) + k[1] * np.exp(etaMat2) + c * (k[0] + k[1]) * np.multiply(np.exp(etaMat1), np.exp(etaMat2))
    ddf = k[0] ** 2 * np.exp(etaMat1) + k[1] ** 2 * np.exp(etaMat2) + c * (k[0] + k[1]) ** 2 * np.multiply(np.exp(etaMat1), np.exp(etaMat2))

    y = 2 * np.divide(np.multiply(f, ddf) - df ** 2, f ** 2)

    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0) # avoid numerical errors far outside of [-1, 2]

    return y


# Plot the exact solution in the time-space domain
def plot_exact_solution():
	x_plot = np.linspace(X_lower, X_upper, 100)
	t_plot = np.linspace(T_lower, T_upper, 40)
	X, T = np.meshgrid(x_plot, t_plot, indexing='ij')
	U = exactKdVTwoSol(x_plot, t_plot)
	plt.contourf(X, T, U, levels=100)
	plt.colorbar()
	plt.xlabel('x')
	plt.ylabel('t')
	plt.title('Exact solution')
	plt.show()

# plot_exact_solution()

# Plot the exact initial condition
def plot_exact_initial_condition():
	x_plot = np.linspace(X_lower, X_upper, 100)
	plt.plot(x_plot, exactKdVTwoSol(x_plot, 0))
	plt.xlabel('x')
	plt.ylabel('u_0')
	plt.title('Exact initial condition')
	plt.show()

# plot_exact_initial_condition()
	

####### Neural network #######
	
# Neural network architecture
# phi = lambda x, w, b: torch.exp(- w ** 2 * torch.abs(torch.sin(torch.pi * (x - b) / L)) ** 2) # gaussian kernel (nonlinear activation function)
# U = lambda x, c, w, b: sum([c[i] * phi(x, w[i], b[i]) for i in range(m)]) # shallow network
	
# v1 = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
# v2 = torch.tensor([[0.1, 0.1], [0.2, 0.2]])
# v3 = torch.tensor([3.0, 3.0])
# print(torch.linalg.norm(v1 - v2, dim=1))
	
# https://auro-227.medium.com/writing-a-custom-layer-in-pytorch-14ab6ac94b77

# Activation function
class PeriodicPhi(nn.Module):

    def __init__(self, d=1, m=1, L=1):
        super(PeriodicPhi, self).__init__()
        self.d = d # input dimension
        self.m = m # number of nodes
        self.L = L
        self.w = torch.nn.Parameter(torch.randn((m))) # random init
        self.b = torch.nn.Parameter(torch.randn((m, d))) # random init

    def forward(self, x):
        # x.shape = (batch, d)
        # b.shape = (m, d)
        x_ext = x.unsqueeze(1) # x_ext.shape = (batch, 1, d)
        # phi = torch.exp(- self.w ** 2 * torch.linalg.norm(torch.sin(np.pi * torch.add(x_ext, - self.b) / self.L), dim=2) ** 2)
        phi = torch.exp(- self.w ** 2 * torch.abs(torch.sin(np.pi * torch.add(x_ext, - self.b) / self.L)).squeeze() ** 2)
        return phi # phi.shape = (batch, m)
	
    def string(self):
        return "Phi(d={}, m={}, L={})".format(self.d, self.m, self.L)


# Test the custom periodic layer
# layer = PeriodicPhi(3, 5)
# x = torch.rand(10, 3)
# print(x)
# y = layer.forward(x)
# print(y.shape)

# Shallow neural network
shallow_net = nn.Sequential(
    PeriodicPhi(d, m, L),
    nn.Linear(m, 1, bias=False)
)

# x = torch.linspace(X_lower, X_upper, 100).reshape(-1, 1)
# y = shallow_net.forward(x)
# plt.plot(x.numpy(), y.detach().numpy())
# plt.show()

print(shallow_net)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(shallow_net.parameters(), lr=gamma, weight_decay=1e-6)


####### Initialization #######

# Select device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move the model to the device
shallow_net.to(device)

# Dataset definition
class CreateDataset(torch.utils.data.Dataset):

    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __getitem__(self, index):
        sample = {
            'point': self.x[index],
            'solution': self.y[index]}
        return sample

    def __len__(self):
        return len(self.x)


def train_epoch(dataloader):

    shallow_net.train()
     
    for batch_idx, data in enumerate(dataloader):
        # Select data for current batch
        x_batch = data['point']
        U_true_batch = data['solution']
        x_batch = x_batch.to(device)
        U_true_batch = U_true_batch.to(device)
        # Forward pass
        U_batch = shallow_net(x_batch) # U.shape = (n, d)
        loss = criterion(U_batch, U_true_batch)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        # Update weights
        optimizer.step()

    return loss.item()


def initialize_parameters():
	
	# Find the initial parameters by fitting a least squares problem
	# Equivalently, train the shallow neural network at time t = 0

    # Each batch should contain n = 1000 points (?)
	
	# Sample points uniformly in [0, 1]
    x = np.random.rand(batch_size * 100, d)

    # Define target
    U_true = exactKdVTwoSol(x, 0)

    # Define dataset and dataloader
    dataset = CreateDataset(x, U_true)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    losses = []

	# Fit the initial condition
    print("Fitting the initial condition...")
    for epoch in range(epochs):
        loss = train_epoch(dataloader)
        losses.append(loss)
        if epoch % 1000 == 0:
            print("	Epoch: {}/{} - Loss: {}".format(epoch, epochs, loss))

    # Plot the loss
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    # Move model to the CPU
    shallow_net.cpu()

    # Plot the true and fitted initial conditions
    x_plot = torch.linspace(X_lower, X_upper, 100).reshape(-1, 1)
    with torch.no_grad():
        U = shallow_net(x_plot)
    plt.plot(x_plot.numpy(), exactKdVTwoSol(x_plot, 0), label='True')
    plt.plot(x_plot.numpy(), U.numpy(), label='Fitted')
    plt.legend()
    plt.show()

    # Compute relative error
    relative_error = torch.linalg.norm(U - exactKdVTwoSol(x_plot, 0)) / torch.linalg.norm(exactKdVTwoSol(x_plot, 0))
    print("Relative error:", relative_error)

# initialize_parameters()


# Periodic boundary conditions?

