import jax
import jax.numpy as jnp
from functools import partial


# @jax.jit
@partial(jax.jit, static_argnums=(0,))
def M_fn(u_fn, theta_flat, x):
    '''
    Assemble the M matrix.
    '''
    U_dtheta = jax.vmap(jax.grad(u_fn), (None, 0))
    u_dth = U_dtheta(theta_flat, x)
    M = jnp.mean(u_dth[:, :, jnp.newaxis] * u_dth[:, jnp.newaxis, :], axis=0)
    return M


# @jax.jit
@partial(jax.jit, static_argnums=(0,1,))
def F_fn(u_fn, rhs, theta_flat, x, t):
    '''
    Assemble the F matrix.
    '''
    U_dtheta = jax.vmap(jax.grad(u_fn), (None, 0))
    u_dth = U_dtheta(theta_flat, x)
    f = rhs(theta_flat, x, t, u_fn) # source term
    F = jnp.mean(u_dth[:, :] * f[:, jnp.newaxis], axis=0)
    return F


# @jax.jit
@partial(jax.jit, static_argnums=(0,1,))
def r_fn(u_fn, rhs, theta_flat, delta_theta_flat, x, t):
    '''
    Compute the local-in-time residual.
    '''
    x = x.reshape(-1, 1)
    U_dtheta = jax.vmap(jax.grad(u_fn), (None, 0))
    r = jnp.dot(U_dtheta(theta_flat, x), delta_theta_flat) - rhs(theta_flat, x, t, u_fn)
    return r.squeeze()