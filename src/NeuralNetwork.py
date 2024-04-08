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


class DeepNetKdV(nn.Module):

    m: int

    @nn.compact
    def __call__(self, x):
        net = nn.Sequential([nn.Dense(features=self.m),
                             nn.activation.sigmoid,
                             nn.Dense(features=self.m),
                             nn.activation.sigmoid,
                             nn.Dense(features=1, use_bias=False)])
        if len(x.shape) == 1:
            return jnp.squeeze(net(x), 0)
        return net(x)
