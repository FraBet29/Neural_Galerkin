import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from typing import Callable


class PeriodicPhi(nn.Module):

    d: int
    m: int
    L: float
    param_init: Callable = nn.initializers.uniform()
    # param_init: Callable = nn.initializers.truncated_normal(lower=-1.0, upper=1.0)
    # param_init: Callable = nn.initializers.constant(1)

    @nn.compact
    def __call__(self, x):
        batch = x.shape[0] # x.shape = (batch, d)
        w = self.param('kernel', self.param_init, (self.m, )) # w.shape = (m, )
        b = self.param('bias', self.param_init, (self.m, self.d)) # b.shape = (m, d)
        x_ext = jnp.expand_dims(x, 1) # x_ext.shape = (batch, 1, d)
        if batch == 1:
            x_ext = jnp.expand_dims(x_ext, 0) # x_ext.shape = (1, 1, d)
        phi = jnp.exp(- w ** 2 * jnp.linalg.norm(jnp.sin(jnp.pi * jnp.add(x_ext, - b) / self.L), axis=2) ** 2)
        if batch == 1:
            phi = jnp.squeeze(phi, 0)
        # phi = jnp.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0) # avoid numerical errors (due to exp)
        return phi # phi.shape = (batch, m)


class ShallowNet(nn.Module):

    d: int
    m: int
    L: float

    @nn.compact
    def __call__(self, x):
        batch = x.shape[0] # x.shape = (batch, d)
        net = nn.Sequential([PeriodicPhi(self.d, self.m, self.L),
                              nn.Dense(features=1, use_bias=False)])
        if batch == 1:
            return jnp.squeeze(net(x), 0)
        return net(x)
