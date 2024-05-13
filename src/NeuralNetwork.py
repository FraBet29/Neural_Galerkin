import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from typing import Callable


class PeriodicPhiKdV(nn.Module):

    m: int
    L: float
    param_init: Callable = nn.initializers.uniform()

    @nn.compact
    def __call__(self, x):

        # Initialize the parameters
        d = x.shape[-1] # input dimension
        w = self.param('kernel', self.param_init, (self.m, )) # w.shape = (m, )
        b = self.param('bias', self.param_init, (self.m, d)) # b.shape = (m, d)

        # TRAIN: x.shape = (d, )
        # EVAL: x.shape = (batch, d)

        def apply_phi(x):
            return jnp.exp(- w ** 2 * jnp.linalg.norm(jnp.sin(jnp.pi * jnp.add(jnp.expand_dims(x, 0), - b) / self.L), axis=1) ** 2)

        # Apply phi to each input
        phi = jax.vmap(apply_phi)(x)

        return phi
    

# This layer is not actually trained
class PeriodicPhiKdVLinear(nn.Module):

    m: int
    L: float
    w: jnp.array
    b: jnp.array

    @nn.compact
    def __call__(self, x):

        def apply_phi(x):
            return jnp.exp(- self.w ** 2 * jnp.linalg.norm(jnp.sin(jnp.pi * jnp.add(jnp.expand_dims(x, 0), - self.b) / self.L), axis=1) ** 2)

        # Apply phi to each input
        phi = jax.vmap(apply_phi)(x)

        return phi


class ShallowNetKdV(nn.Module):

    m: int
    L: float

    @nn.compact
    def __call__(self, x):
        net = nn.Sequential([PeriodicPhiKdV(self.m, self.L),
                             nn.Dense(features=1, use_bias=False)])
        if len(x.shape) == 1:
            return jnp.squeeze(net(x), 0)
        return net(x)


class ShallowNetKdVLinear(nn.Module):

    m: int
    L: float
    w: jnp.array
    b: jnp.array

    @nn.compact
    def __call__(self, x):
        net = nn.Sequential([PeriodicPhiKdVLinear(self.m, self.L, self.w, self.b),
                             nn.Dense(features=1, use_bias=False)])
        if len(x.shape) == 1:
            return jnp.squeeze(net(x), 0)
        return net(x)


# class DeepNetKdV(nn.Module):

#     m: int

#     @nn.compact
#     def __call__(self, x):
#         net = nn.Sequential([nn.Dense(features=self.m),
#                              nn.activation.sigmoid,
#                              nn.Dense(features=self.m),
#                              nn.activation.sigmoid,
#                              nn.Dense(features=1, use_bias=False)])
#         if len(x.shape) == 1:
#             return jnp.squeeze(net(x), 0)
#         return net(x)


class PeriodicPhiAC(nn.Module):

    m: int
    L: float
    param_init: Callable = nn.initializers.truncated_normal(stddev=1.0)
    # param_init: Callable = nn.initializers.uniform()
    # param_init: Callable = nn.initializers.constant(1)

    @nn.compact
    def __call__(self, x):

        d = x.shape[-1] # input dimension

        # w = self.param('kernel', self.param_init, (self.m, d)) # w.shape = (m, d)
        # b = self.param('bias', self.param_init, (d, )) # b.shape = (d, )

        a = self.param('a', self.param_init, (self.m, d))
        b = self.param('b', self.param_init, (self.m, d))
        c = self.param('c', self.param_init, (self.m, d))

        def apply_phi(x):
            # return w @ jnp.sin(2 * jnp.pi * (x - b) / self.L)
            return jnp.sum(a * jnp.cos((2 * jnp.pi / self.L) * x + b) + c, axis=1)

        # Apply phi to each input
        phi = jax.vmap(apply_phi)(x)

        return phi
    

# class Rational(nn.Module):
#     """
#     Rational activation function
#     Ref.: Nicolas Boullé, Yuji Nakatsukasa, and Alex Townsend,
#           Rational neural networks,
#           arXiv preprint arXiv:2004.01902 (2020).
#     """
#     alpha_init = lambda *args: jnp.array([1.1915, 1.5957, 0.5, 0.0218])
#     beta_init = lambda *args: jnp.array([2.383, 0.0, 1.0])
    
#     @nn.compact
#     def __call__(self, x):
#         alpha = self.param('alpha', self.alpha_init)
#         beta = self.param('beta', self.beta_init)
#         return jnp.polyval(alpha, x) / jnp.polyval(beta, x)


class DeepNetAC(nn.Module):

    m: int
    l: int
    L: float

    @nn.compact
    def __call__(self, x):
        layers = [PeriodicPhiAC(self.m, self.L), nn.activation.tanh]
        for _ in range(self.l-1):
            layers.append(nn.Dense(self.m))
            layers.append(nn.activation.tanh)
        layers.append(nn.Dense(features=1, use_bias=False))
        net = nn.Sequential(layers)
        if len(x.shape) == 1:
            return jnp.squeeze(net(x), 0)
        return net(x)
