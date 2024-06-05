import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from typing import Callable
from flax.training.train_state import TrainState
import optax
from flax.training import train_state
import haiku as hk

# JAX does not have data loaders (?)

# TEST 1

def custom_init(key, shape, dtype=jnp.float32):
    return jax.random.uniform(key, shape, dtype, minval=-1.0, maxval=1.0)

class Linear(nn.Module):
    
    features_in: int
    features_out: int
    # param_init: Callable = nn.initializers.truncated_normal()
    param_init: Callable = custom_init

    @nn.compact
    def __call__(self, x):
        w = self.param('w', self.param_init, (2, 1))
        b = self.param('b', self.param_init, (1, 1))
        y = jnp.dot(w, x) + b
        return y.squeeze() # returns a scalar
	

n = 1
key1, key2 = jax.random.split(jax.random.key(0), 2)
x = jax.random.uniform(key1, (n, 2))
# print(x)

net = Linear(2, 1)
params = net.init(key2, x)
# print('Parameters:\n', jax.tree_util.tree_map(jnp.shape, flax.core.unfreeze(params)))
# print(jax.tree_map(lambda x: x.shape, params))
# print(params)

u = net.apply(params, x)
# print(u)

# TEST 2

# https://dm-haiku.readthedocs.io/en/latest/notebooks/flax.html

# def build_net():
#     def net(x):
#         layers = [Linear1D()]
#         y = nn.Sequential(layers)(x)
#         return jnp.squeeze(y)
#     return net


class Linear1D(nn.Module):
    
    # features_in: int
    # features_out: int
    # param_init: Callable = nn.initializers.truncated_normal()
    param_init: Callable = custom_init

    @nn.compact
    def __call__(self, x):
        w = self.param('w', self.param_init, (1, 1))
        b = self.param('b', self.param_init, (1, 1))
        y = w * x ** 2 + b
        return y.squeeze() # returns a scalar

# https://flax.readthedocs.io/en/latest/developer_notes/lift.html

class VmapLinear1D(nn.Module):
  
  @nn.compact
  def __call__(self, xs):
    l1D = Linear1D(parent=None)
    init_fn = lambda rng, xs: jax.vmap(l1D.init, in_axes=0)(jax.random.split(rng, xs.shape[0]), xs)['params']
    apply_fn = jax.vmap(l1D.apply, in_axes=0)
    l1D_params = self.param('l1D', init_fn, xs)
    return apply_fn({'l1D': l1D_params}, xs)


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


def init_net(net, key, dim):
    # trans = hk.without_apply_rng(hk.transform(net))
    # net_apply, net_init = trans.apply, trans.init
    net_apply, net_init = net.apply, net.init # PURE FUNCTIONS?
    print(net_apply)
    print(net_init)
    theta_init = net_init(key, jnp.zeros(dim))
    print(jax.tree_map(lambda x: x, theta_init))
    theta_init_flat, unravel = jax.flatten_util.ravel_pytree(theta_init) # ravel (flatten) a pytree of arrays down to a 1D array
    # theta_init_flat: 1D array representing the flattened and concatenated leaf values
    # unravel: callable for unflattening a 1D vector of the same length back to a pytree of the same structure as the input pytree
    u_scalar = unraveler(net_apply, unravel)
    return u_scalar, theta_init_flat, unravel


# net = build_net()
net = Linear1D()
# print(net)

# net_fn = VmapLinear1D()
# print(net_fn)

u_scalar, theta_init, unravel = init_net(net, jax.random.key(0), 1)
# print(u_scalar)
# print(theta_init)
# print(unravel)

# Take gradient and then squeeze
def gradsqz(f, *args, **kwargs):
    '''
    Function taken from: https://github.com/julesberman/RSNG/blob/main/allen_cahn.ipynb
    '''
    return lambda *fargs, **fkwargs: jnp.squeeze(jax.grad(f, *args, **kwargs)(*fargs, **fkwargs))


# def f(x, y):
#     return x ** 3 + y ** 2

# print(gradsqz(gradsqz(f))(1.0, 1.0))

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


t = 0
x = jnp.expand_dims(jnp.linspace(0, 1, 5), axis=-1)
source = rhs(t, theta_init, x)
# print(source)

# https://github.com/google/flax/issues/283
