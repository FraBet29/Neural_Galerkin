import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from typing import Callable
import matplotlib.pyplot as plt
from jax import flatten_util
import optax
from flax.training import train_state
from torch.utils.data import Dataset, DataLoader
from flax import serialization
from matplotlib.animation import FuncAnimation


##################################################################################################################
############################################## AC equation #######################################################
##################################################################################################################
    
####### Problem data #######

# Space-time domain
X_lower = 0
X_upper = 2 * jnp.pi
T = 4
dt = 0.01

# PDE parameters
epsilon = 5e-2
a = lambda x, t: (1.05 + t * jnp.sin(x)).squeeze()

# NN architecture
d = 1 # input dimension
m = 2 # nodes
l = 3 # hidden layers
L = 0.5 # parameter for the gaussian kernel
n = 1000 # samples
batch_size = 5000 # batch size (since n = 1000, we will take the whole dataset?)
epochs = 5000 # number of epochs
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
        w = self.param('kernel', self.param_init, self.m) # w.shape = (m, )
        b = self.param('bias', self.param_init, (self.m, self.d)) # b.shape = (m, d)
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
                              nn.activation.tanh,
                              nn.Dense(features=m),
                              nn.activation.tanh,
                              nn.Dense(features=m),
                              nn.activation.tanh,
                              nn.Dense(features=1, use_bias=False)])
        if batch == 1:
            return jnp.squeeze(net(x), 0)
        return net(x)


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
    # print(jax.tree_map(lambda x: x, theta_init))
    theta_init_flat, unravel = flatten_util.ravel_pytree(theta_init) # ravel (flatten) a pytree of arrays down to a 1D array
    # theta_init_flat: 1D array representing the flattened and concatenated leaf values
    # unravel: callable for unflattening a 1D vector of the same length back to a pytree of the same structure as the input pytree
    u_scalar = unraveler(net_apply, unravel)
    return u_scalar, theta_init, theta_init_flat, unravel


# periodic_phi = PeriodicPhi(d, m, L)
# u_scalar, theta_init, unravel = init_net(periodic_phi, key2, x)
deep_net = DeepNet(d, m, L)
print(deep_net)

# psi = lambda x, w, b: -2 * w ** 2 * jnp.sin(jnp.pi * (x - b) / L) * jnp.cos(jnp.pi * (x - b) / L) * jnp.pi / L
# phi_prime = lambda x, w, b: phi(x, w, b) * psi(x, w, b)
# phi_second = lambda x, w, b: phi_prime(x, w, b) * psi(x, w, b) + phi(x, w, b) * (-2 * w ** 2) * (jnp.cos(jnp.pi * (x - b) / L) ** 2 - jnp.sin(jnp.pi * (x - b) / L) ** 2) * (jnp.pi / L) ** 2
# u_check = lambda x, w, b, c: c[0] * phi_second(x, w[0], b[0]) + c[1] * phi_second(x, w[1], b[1]) 
# print(u_check(x[0], jnp.ones(2), jnp.ones(2), jnp.array([-0.2714497, -0.7051541])))


####### Initialization #######

# Dataset definition
class CreateDataset(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        sample = (self.x[index], self.y[index])
        return sample

    def __len__(self):
        return len(self.x)


# def collate_fn(batch):
#     if isinstance(batch[0], jnp.ndarray):
#         return jnp.stack(batch)
#     elif isinstance(batch[0], tuple):
#         return tuple(collate_fn(samples) for samples in zip(*batch))
#     else:
#         return jnp.asarray(batch)
    

# # @jax.jit
# def mse_loss(model, params, x_batched, y_batched):
#     '''
#     Function taken from: https://flax.readthedocs.io/en/latest/guides/flax_fundamentals/flax_basics.html
#     '''
#     # Define the squared loss for a single pair (x, y)
#     def squared_error(x, y):
#         pred = model.apply(params, x)
#         return jnp.inner(y - pred, y - pred) / 2.0
#     # Vectorize the previous to compute the average of the loss on all samples
#     return jnp.mean(jax.vmap(squared_error)(x_batched, y_batched), axis=0)


def make_mse_loss(model, xs, ys):

    def mse_loss(params):

        # Define the squared loss for a single pair (x,y)
        def squared_error(x, y):
            pred = model.apply(params, x)
            # Inner because 'y' could have in general more than 1 dims
            return jnp.inner(y - pred, y - pred) / 2.0

        # Batched version via vmap
        return jnp.mean(jax.vmap(squared_error)(xs, ys), axis=0)

    return jax.jit(mse_loss)  # and finally we jit the result (mse_loss is a pure function)


# https://huggingface.co/blog/afmck/flax-tutorial

# # @jax.jit
# def train_step(model, state: train_state.TrainState, batch: jnp.ndarray):
    
#     x_batched, y_batched = batch

#     def loss_fn(params):
#         preds = state.apply_fn({'params': params}, x_batched)
#         loss = mse_loss(model, {'params': params}, x_batched, y_batched)
#         return loss, preds

#     gradient_fn = jax.value_and_grad(loss_fn, has_aux=True)
#     _, grads = gradient_fn(state.params)
#     state = state.apply_gradients(grads=grads)
#     metrics = compute_metrics(model, state.params, x_batched, y_batched)
#     print("Metrics:", metrics)
#     return state, metrics


def init_neural_galerkin(deep_net):
	
	# Find the initial parameters by fitting a least squares problem
	# Equivalently, train the shallow neural network at time t = 0

    # Each batch should contain n = 1000 points (?)

    # Generate random keys and input data
    key1, key2 = jax.random.split(jax.random.key(0))
    x_init = jax.random.uniform(key1, (batch_size, d), minval=X_lower, maxval=X_upper)

    # Define target
    u_true = u_0(x_init)
	
    # Initialize the model
    # state, _, theta_init, _, _ = init_train_state(deep_net, key2, x)
    u_scalar, theta_init, theta_init_flat, unravel = init_net(deep_net, key2, x_init)
    
    # Define dataset and dataloader
    # dataset = CreateDataset(x_init, u_true)
    # dataloader = DataLoader(dataset, batch_size=n, collate_fn=collate_fn, shuffle=True)

    # Define the optimizer
    opt = optax.adam(learning_rate=gamma)
    opt_state = opt.init(theta_init)

    # Define the loss function
    mse_loss = make_mse_loss(deep_net, x_init, u_true)
    value_and_grad_fn = jax.value_and_grad(mse_loss)

    # Define a TrainState
    # state = train_state.TrainState.create(apply_fn=deep_net.apply, tx=opt, params=theta_init['params'])

    losses = []

    # Fitting the initial condition
    for epoch in range(epochs):
        loss, grads = value_and_grad_fn(theta_init)
        updates, opt_state = opt.update(grads, opt_state)
        theta_init = optax.apply_updates(theta_init, updates)
        losses.append(loss)

        if epoch % 100 == 0:
            err = jnp.linalg.norm(u_true - deep_net.apply(theta_init, x_init)) / jnp.linalg.norm(u_true)
            print(f'epoch {epoch}, loss = {loss}, error = {err}')

    # Plot the loss
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    # Plot the true and fitted initial conditions
    x_plot = jnp.linspace(X_lower, X_upper, 100).reshape(-1, 1)
    u_pred = deep_net.apply(theta_init, x_plot)
    plt.plot(x_plot, u_0(x_plot), label='True')
    plt.plot(x_plot, u_pred, label='Fitted')
    plt.legend()
    plt.show()

    # Compute relative error
    relative_error = jnp.linalg.norm(u_pred - u_0(x_plot)) / jnp.linalg.norm(u_0(x_plot))
    print("Relative error:", relative_error)

    return theta_init


# theta_init = init_neural_galerkin(deep_net)

# Save the initial parameters
# jnp.save('theta_init.npy', theta_init)


####### Time evolution #######

deep_net = DeepNet(d, m, L)

# Initialize the model
key1, key2 = jax.random.split(jax.random.key(0))
x = jax.random.uniform(key1, (n, d), minval=X_lower, maxval=X_upper)
u_scalar, theta, theta_flat, unravel = init_net(deep_net, key2, x)

# Reload the initial parameters
theta_init = jnp.load('theta_init.npy', allow_pickle=True).item()

# bytes_output = serialization.to_bytes(theta_init)
# serialization.from_bytes(theta, bytes_output)
# serialization.from_state_dict(theta, theta_init) # does not work (keys in different order)

theta['params']['PeriodicPhi_0'] = theta_init['params']['PeriodicPhi_0']
theta['params']['Dense_0'] = theta_init['params']['Dense_0']
theta['params']['Dense_1'] = theta_init['params']['Dense_1']
theta['params']['Dense_2'] = theta_init['params']['Dense_2']

# x_plot = jnp.linspace(X_lower, X_upper, 100).reshape(-1, 1)
# u_pred = deep_net.apply(theta, x_plot)
# plt.plot(x_plot, u_0(x_plot), label='True')
# plt.plot(x_plot, u_pred, label='Fitted')
# plt.legend()
# plt.show()


# Take gradient and then squeeze
def gradsqz(f, *args, **kwargs):
    '''
    Function taken from: https://github.com/julesberman/RSNG/blob/main/allen_cahn.ipynb
    '''
    return lambda *fargs, **fkwargs: jnp.squeeze(jax.grad(f, *args, **kwargs)(*fargs, **fkwargs))

# Batch the function over X points
U = jax.vmap(u_scalar, (None, 0)) # jax.vmap(fun, in_axes)

# Derivative w.r.t. theta
U_dtheta = jax.vmap(jax.grad(u_scalar), (None, 0))

# Spatial derivatives
U_ddx = jax.vmap(gradsqz(gradsqz(u_scalar, 1), 1), (None, 0))

# Source term for the AC equation
def rhs(t, theta, x):
    u = U(theta, x)
    u_xx = U_ddx(theta, x)
    return epsilon * u_xx - a(x, t) * (u - u ** 3)


def neural_galerkin(deep_net, theta):

    solution = []

    t = 0

    while t < T:

        if int(t / dt) % 20 == 0:
            print(f'Time: {t:.2f}')
        
        # Sample points in the spatial domain
        x = jax.random.uniform(jax.random.key(int(t * 1e4)), (n, d), minval=X_lower, maxval=X_upper)
        x_plot = jnp.linspace(X_lower, X_upper, 100).reshape(-1, 1) # for plotting

        # Flatten the parameters
        theta_flat = jax.flatten_util.ravel_pytree(theta)[0]
        
        # Approximate M and F
        u_dth = U_dtheta(theta_flat, x)
        M = jnp.mean(u_dth[:, :, jnp.newaxis] * u_dth[:, jnp.newaxis, :], axis=0)
        f = rhs(t, theta, x) # source term
        F = jnp.mean(u_dth * f[:, jnp.newaxis], axis=0)

        # Update the parameters
        theta_flat = jnp.linalg.solve(M, F)
        # TODO: IMPLICIT EULER

        # Update the solution
        u_scalar = unraveler(deep_net.apply, unravel)

        # Save current solution for plotting
        theta = unravel(theta_flat)
        u = U(theta, x_plot)
        solution.append(u)

        t = t + dt

    return solution


solution = neural_galerkin(deep_net, theta)