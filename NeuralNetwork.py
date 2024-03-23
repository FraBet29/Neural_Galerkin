import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from typing import Callable


class PeriodicPhi(nn.Module):

    m: int
    L: float
    # param_init: Callable = nn.initializers.uniform()
    param_init: Callable = nn.initializers.truncated_normal(lower=0.0, upper=1.0)

    @nn.compact
    def __call__(self, x):

        # Initialize the parameters
        d = x.shape[-1] # input dimension
        w = self.param('kernel', self.param_init, (self.m, )) # w.shape = (m, )
        b = self.param('bias', self.param_init, (self.m, d)) # b.shape = (m, d)

        # TRAIN: x.shape = (d, )
        # EVAL: x.shape = (batch, d)

        if len(x.shape) == 1:
            phi = jnp.exp(- w ** 2 * jnp.linalg.norm(jnp.sin(jnp.pi * jnp.add(jnp.expand_dims(x, 0), - b) / self.L), axis=1) ** 2)
        else:
            phi = jnp.exp(- w ** 2 * jnp.linalg.norm(jnp.sin(jnp.pi * jnp.add(jnp.expand_dims(x, 1), - b) / self.L), axis=2) ** 2)

        return phi


class ShallowNet(nn.Module):

    m: int
    L: float

    @nn.compact
    def __call__(self, x):
        net = nn.Sequential([PeriodicPhi(self.m, self.L),
                              nn.Dense(features=1, use_bias=False)])
        if len(x.shape) == 1:
            return jnp.squeeze(net(x), 0)
        return net(x)


class DeepNet(nn.Module):

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