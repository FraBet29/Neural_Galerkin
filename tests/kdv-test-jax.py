# import numpy as np
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import matplotlib.pyplot as plt
from typing import Callable


####### Automatic differentiation #######

def dx(f):
	return jax.grad(f)

def ddx(f):
	return jax.grad(jax.grad(f))

# Test automatic differentiation
# def f(x, t):
# 	return x ** 4 + t ** 3

# x = 2.0
# t = 1.0

# print("df_x(x, t) =", dx(f)(x, t))
# print("ddf_x(f, x, t) =", ddx(f)(x, t))


####### Problem data #######

# Space-time domain
X_lower = -10
X_upper = 20
T_lower = 0
T_upper = 4

# NN architecture
d = 2 # input dimension
m = 10 # nodes
L = 60 # parameter for the gaussian kernel
# n = 1000 # samples
n = 100
batch = 10 ** 5 # batch size (since n = 1000, we will take the whole dataset?)
# epochs = 10 ** 5 # number of epochs
epochs = 100
gamma = 0.1 # learning rate


####### Exact solution #######
# k1 = 1
# k2 = 5 ** (1/2)
# eta1_0 = 0
# eta2_0 = 10.73
# eta1 = lambda x, t: k1 * x - k1 ** 3 * t + eta1_0
# eta2 = lambda x, t: k2 * x - k2 ** 3 * t + eta2_0
# g = lambda x, t: 1 + jnp.exp(eta1(x, t)) + jnp.exp(eta2(x, t)) + jnp.exp(eta1(x, t) + eta2(x, t)) * ((k1 - k2) / (k1 + k2)) ** 2
# # log_g = lambda x, t: 2 * eta1(x, t) + 2 * eta2(x, t) + 2 * jnp.log((k1 - k2) / (k1 + k2))
# log_g = lambda x, t: jnp.log(g(x, t))
# u = lambda x, t: 2 * ddx(log_g)(x, t)
# u_0 = lambda x: u(x, 0)

def exactKdVTwoSol(x, t):
    '''
    Function taken from https://github.com/pehersto/ng (exactKdV.py)
    '''

    k = jnp.asarray([1., jnp.sqrt(5.)])
    eta = jnp.asarray([0., 10.73])
    t = jnp.asarray(t)

    etaMat1 = k[0] * x.reshape((-1, 1)) - k[0] ** 3 * t.reshape((1, -1)) + eta[0]
    etaMat2 = k[1] * x.reshape((-1, 1)) - k[1] ** 3 * t.reshape((1, -1)) + eta[1]
    c = ((k[0] - k[1]) / (k[0] + k[1]) )** 2

    f = 1. + jnp.exp(etaMat1) + jnp.exp(etaMat2) + jnp.multiply(jnp.exp(etaMat1), jnp.exp(etaMat2) * c)
    df = k[0] * jnp.exp(etaMat1) + k[1] * jnp.exp(etaMat2) + c * (k[0] + k[1]) * jnp.multiply(jnp.exp(etaMat1), jnp.exp(etaMat2))
    ddf = k[0] ** 2 * jnp.exp(etaMat1) + k[1] ** 2 * jnp.exp(etaMat2) + c * (k[0] + k[1]) ** 2 * jnp.multiply(jnp.exp(etaMat1), jnp.exp(etaMat2))

    y = 2 * jnp.divide(jnp.multiply(f, ddf) - df ** 2, f ** 2)

    y = jnp.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0) # avoid numerical errors far outside of [-1, 2]

    return y


# Plot the exact solution in the time-space domain
def plot_exact_solution():
	x_plot = jnp.linspace(X_lower, X_upper, 100)
	t_plot = jnp.linspace(T_lower, T_upper, 40)
	X, T = jnp.meshgrid(x_plot, t_plot, indexing='ij')
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
	x_plot = jnp.linspace(X_lower, X_upper, 100)
	plt.plot(x_plot, exactKdVTwoSol(x_plot, 0))
	plt.xlabel('x')
	plt.ylabel('u_0')
	plt.title('Exact initial condition')
	plt.show()

# plot_exact_initial_condition()
	

####### Neural network #######
	
# Neural network architecture
      
class PeriodicPhi(nn.Module):
    
    features: int
    L: float
    param_init: Callable = nn.initializers.truncated_normal()

    @nn.compact
    def __call__(self, x):
        d, m = x.shape[-1], self.features
        w = self.param('w', self.param_init, (m, 1))
        b = self.param('b', self.param_init, (m, d))
        c = self.param('c', self.param_init, (m, 1))
        print(x.shape)
        print(w.shape)
        print(b.shape)
        print(c.shape)
        return jnp.sum(c * jnp.exp(- w ** 2 * jnp.linalg.norm(jnp.sin(jnp.pi * (x - b) / self.L), axis=1) ** 2), axis=1)


key1, key2 = jax.random.split(jax.random.key(0), 2)
x = jax.random.uniform(key1, (n, d))

Phi = PeriodicPhi(m, L)
params = Phi.init(key2, x)
print('Parameters:\n', jax.tree_util.tree_map(jnp.shape, flax.core.unfreeze(params)))
	

# Shallow neural network
shallow_net = nn.Sequential(
    Phi(d, m, L),
    nn.Linear(m, 1, bias=False)
)

print(shallow_net)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(shallow_net.parameters(), lr=gamma)


####### Initialization #######
def initialize_parameters():
	
	# c = torch.zeros(m)
	# w = torch.zeros(m)
	# b = torch.zeros(m)
	
	# Find the initial parameters by fitting a least squares problem
	# Equivalently, train the shallow neural network at time t = 0
	
	# Sample points uniformly in [0, 1]
	x = torch.rand(n, d)

	# Fit the initial condition
	print("Fitting the initial condition...")
	for epoch in range(epochs):
		if epoch % 10 == 0:
			print("	Epoch {}/{}".format(epoch, epochs))
		# Forward pass
		U = shallow_net(x)
		loss = criterion(U, u_0(x))
		# Backward pass
		optimizer.zero_grad()
		loss.backward()
		# Update weights
		optimizer.step()

	# Print the true and fitted initial conditions
	x_plot = torch.linspace(X_lower, X_upper, 100)
	with torch.no_grad():
		U = shallow_net(x)
	plt.plot(x_plot.numpy(), u_0(x_plot), label='True')
	plt.plot(x_plot.numpy(), U.numpy(), label='Fitted')
	plt.legend()
	plt.show()

# initialize_parameters()


# Periodic boundary conditions?
