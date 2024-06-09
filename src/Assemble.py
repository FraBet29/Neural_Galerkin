import jax
import jax.numpy as jnp
from functools import partial


# @jax.jit
@partial(jax.jit, static_argnums=(0, ))
def M_fn(u_fn, theta_flat, x, reg=0):
    '''
    Assemble the M matrix.
    '''
    U_dtheta = jax.vmap(jax.grad(u_fn), (None, 0))
    u_dth = U_dtheta(theta_flat, x)
    u_dth = jnp.where(jnp.isnan(u_dth), 0.0, u_dth)
    M = jnp.mean(u_dth[:, :, jnp.newaxis] * u_dth[:, jnp.newaxis, :], axis=0)
    M += reg * jnp.eye(M.shape[0]) # regularization
    return M


# @jax.jit
@partial(jax.jit, static_argnums=(0, 1, ))
def F_fn(u_fn, rhs, theta_flat, x, t):
    '''
    Assemble the F matrix.
    '''
    U_dtheta = jax.vmap(jax.grad(u_fn), (None, 0))
    u_dth = U_dtheta(theta_flat, x)
    u_dth = jnp.where(jnp.isnan(u_dth), 0.0, u_dth)
    f = rhs(theta_flat, x, t, u_fn) # source term
    F = jnp.mean(u_dth[:, :] * f[:, jnp.newaxis], axis=0)
    return F


@partial(jax.jit, static_argnums=(0, ))
def J_fn(u_fn, theta_flat, x):
    '''
    Assemble the Jacobian matrix.
    '''
    U_dtheta = jax.vmap(jax.grad(u_fn), (None, 0))
    u_dth = U_dtheta(theta_flat, x)
    u_dth = jnp.where(jnp.isnan(u_dth), 0.0, u_dth)
    return u_dth


# @jax.jit
@partial(jax.jit, static_argnums=(0, 1, ))
def r_fn(u_fn, rhs, theta_flat, delta_theta_flat, x, t):
    '''
    Compute the local-in-time residual.
    '''
    x = x.reshape(-1, 1)
    U_dtheta = jax.vmap(jax.grad(u_fn), (None, 0))
    u_dth = U_dtheta(theta_flat, x)
    u_dth = jnp.where(jnp.isnan(u_dth), 0.0, u_dth)
    r = jnp.dot(u_dth, delta_theta_flat) - rhs(theta_flat, x, t, u_fn)
    return r.squeeze()


@partial(jax.jit, static_argnums=(0, 1, ))
def r_loss(u_fn, rhs, theta_flat, delta_theta_flat, x, t):
    '''
    Compute the square norm of the residual.
    '''
    r = r_fn(u_fn, rhs, theta_flat, delta_theta_flat, x, t)
    return jnp.linalg.norm(r) ** 2


# @partial(jax.jit, static_argnums=(0, 1, ))
def assemble_weighted(u_fn, rhs, theta_flat, x, t, w_fn, problem_data, store):
    '''
    Assemble the weighted M and F matrices.
    '''
    w, u_dth_ort, B_GS = w_fn(x.squeeze(), store) # weights, orthonormal basis, Gram-Schmidt change of basis

    M = jnp.mean(w[:, jnp.newaxis, jnp.newaxis] * (u_dth_ort[:, :, jnp.newaxis] * u_dth_ort[:, jnp.newaxis, :]), axis=0)

    f = rhs(theta_flat, x, t, u_fn) # source term
    F = jnp.mean(w[:, jnp.newaxis] * (u_dth_ort[:, :] * f[:, jnp.newaxis]), axis=0)

    print('cond(B_GS) =', jnp.linalg.cond(B_GS))
    print('cond(M_opt) =', jnp.linalg.cond(M))

    # Solve the weighted least squares problem (Gv = d)
    tau = jnp.linalg.lstsq(M, F)[0]

    # Recover the original coefficients
    theta_flat = jnp.linalg.lstsq(B_GS.T, tau)[0]

    return theta_flat
