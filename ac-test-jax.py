import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from typing import Callable
import matplotlib.pyplot as plt
from jax import flatten_util

##################################################################################################################
############################################## AC equation #######################################################
##################################################################################################################
    
####### Problem data #######

# Space-time domain
X_lower = 0
X_upper = 2 * jnp.pi
T_lower = 0
T_upper = 6

# PDE parameters
epsilon = 5 * 10 ** (-2)
# a = lambda x, t: 1.05 + t * jnp.sin(x)

# NN architecture
d = 1 # input dimension
m = 2 # nodes
l = 3 # hidden layers
L = 0.5 # parameter for the gaussian kernel
n = 1000 # samples
batch_size = 10 ** 5 # batch size (since n = 1000, we will take the whole dataset?)
epochs = 10 ** 4 # number of epochs
gamma = 0.1 # learning rate

# Initial condition
phi = lambda x, w, b: jnp.exp(- w ** 2 * jnp.abs(jnp.sin(jnp.pi * (x - b) / L)) ** 2)
u_0 = lambda x: phi(x, jnp.sqrt(10), 0.5) - phi(x, jnp.sqrt(10), 4.4)

# Plot the exact initial condition
def plot_exact_initial_condition():
	x_plot = jnp.linspace(X_lower, X_upper, 100)
	plt.plot(x_plot, u_0(x_plot))
	plt.xlabel('x')
	plt.ylabel('u_0')
	plt.title('Exact initial condition')
	plt.show()

# plot_exact_initial_condition()
	

####### Neural network #######

# Activation function
class PeriodicPhi(nn.Module):

    d: int
    m: int
    L: float
    # param_init: Callable = nn.initializers.truncated_normal()
    param_init: Callable = nn.initializers.constant(1)

    @nn.compact
    def __call__(self, x):
        batch = x.shape[0] # x.shape = (batch, d)
        w = self.param('w', self.param_init, self.m) # w.shape = (m, )
        b = self.param('b', self.param_init, (self.m, self.d)) # b.shape = (m, d)
        x_ext = jnp.expand_dims(x, 1) # x_ext.shape = (batch, 1, d)
        if batch == 1:
            x_ext = jnp.expand_dims(x_ext, 0) # x_ext.shape = (1, 1, d)
        phi = jnp.exp(- w ** 2 * jnp.linalg.norm(jnp.sin(jnp.pi * jnp.add(x_ext, - b) / self.L), axis=2) ** 2)
        if batch == 1:
            phi = jnp.squeeze(phi, 0)
        return phi # phi.shape = (batch, m)


class DeepNet(nn.Module):
  
    d: int
    m: int
    L: float
  
    @nn.compact
    def __call__(self, x):
        batch = x.shape[0] # x.shape = (batch, d)
        net = nn.Sequential([PeriodicPhi(self.d, self.m, self.L),
                            #   nn.Tanh(),
                            #   nn.Dense(features=m),
                            #   nn.Tanh(),
                            #   nn.Dense(features=m),
                            #   nn.Tanh(),
                              nn.Dense(features=1, use_bias=False)])
        if batch == 1:
            return jnp.squeeze(net(x), 0)
        return net(x)


# Take gradient and then squeeze
def gradsqz(f, *args, **kwargs):
    '''
    Function taken from: https://github.com/julesberman/RSNG/blob/main/allen_cahn.ipynb
    '''
    return lambda *fargs, **fkwargs: jnp.squeeze(jax.grad(f, *args, **kwargs)(*fargs, **fkwargs))


def unraveler(f, unravel, axis=0):
    '''
    Function taken from: https://github.com/julesberman/RSNG/blob/main/rsng/dnn.py
    '''
    def wrapper(*args, **kwargs):
        val = args[axis]
        if (type(val) != dict):
            args = list(args)
            args[axis] = unravel(val) 
            args = tuple(args)
        return f(*args, **kwargs)
    return wrapper


def init_net(net, key, x):
    net_apply, net_init = net.apply, net.init # PURE FUNCTIONS?
    theta_init = net_init(key, x)
    print(jax.tree_map(lambda x: x, theta_init))
    theta_init_flat, unravel = flatten_util.ravel_pytree(theta_init) # ravel (flatten) a pytree of arrays down to a 1D array
    # theta_init_flat: 1D array representing the flattened and concatenated leaf values
    # unravel: callable for unflattening a 1D vector of the same length back to a pytree of the same structure as the input pytree
    u_scalar = unraveler(net_apply, unravel)
    return u_scalar, theta_init_flat, unravel


key1, key2 = jax.random.split(jax.random.key(0))
x = jax.random.normal(key1, (10, d)) # Dummy input data
# periodic_phi = PeriodicPhi(d, m, L)
# u_scalar, theta_init, unravel = init_net(periodic_phi, key2, x)
periodic_phi = deep_net = DeepNet(d, m, L)
u_scalar, theta_init, unravel = init_net(deep_net, key2, x)


# Batch the function over X points
U = jax.vmap(u_scalar, (None, 0)) # jax.vmap(fun, in_axes)

# Derivative w.r.t. theta
U_dtheta = jax.vmap(jax.grad(u_scalar), (None, 0))

# Spatial derivatives
U_ddx = jax.vmap(gradsqz(gradsqz(u_scalar, 1), 1), (None, 0))


# Source term for the AC equation
def rhs(t, theta, x):
    a = lambda x, t: 1.05 + t * jnp.sin(x)
    u = U(theta, x)
    print(u)
    u_xx = U_ddx(theta, x)
    print(u_xx)
    # return (5e-2) * u_xx + a(x, t) * (u - u ** 3)
    return (5e-2) * u_xx + (u - u ** 3)


# print(rhs(0, theta_init, x))

# psi = lambda x, w, b: -2 * w ** 2 * jnp.sin(jnp.pi * (x - b) / L) * jnp.cos(jnp.pi * (x - b) / L) * jnp.pi / L
# phi_prime = lambda x, w, b: phi(x, w, b) * psi(x, w, b)
# phi_second = lambda x, w, b: phi_prime(x, w, b) * psi(x, w, b) + phi(x, w, b) * (-2 * w ** 2) * (jnp.cos(jnp.pi * (x - b) / L) ** 2 - jnp.sin(jnp.pi * (x - b) / L) ** 2) * (jnp.pi / L) ** 2
# u_check = lambda x, w, b, c: c[0] * phi_second(x, w[0], b[0]) + c[1] * phi_second(x, w[1], b[1]) 
# print(u_check(x[0], jnp.ones(2), jnp.ones(2), jnp.array([-0.2714497, -0.7051541])))








# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(deep_net.parameters(), lr=gamma, weight_decay=1e-6)


####### Initialization #######

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

    deep_net.train()
     
    for batch_idx, data in enumerate(dataloader):
        # Select data for current batch
        x_batch = data['point']
        U_true_batch = data['solution']
        x_batch = x_batch.to(device)
        U_true_batch = U_true_batch.to(device)
        # Forward pass
        U_batch = deep_net(x_batch) # U.shape = (n, d)
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
    x = jnp.random.rand(n * 1, d)

    # Define target
    U_true = u_0(x)

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
    deep_net.cpu()

    # Plot the true and fitted initial conditions
    x_plot = torch.linspace(X_lower, X_upper, 100).reshape(-1, 1)
    with torch.no_grad():
        U = deep_net(x_plot)
    plt.plot(x_plot.numpy(), u_0(x_plot), label='True')
    plt.plot(x_plot.numpy(), U.numpy(), label='Fitted')
    plt.legend()
    plt.show()

    # Compute relative error
    relative_error = torch.linalg.norm(U - u_0(x_plot)) / torch.linalg.norm(u_0(x_plot))
    print("Relative error:", relative_error)

# initialize_parameters()