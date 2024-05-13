import jax
import jax.numpy as jnp
from functools import partial


# @jax.jit
@partial(jax.jit, static_argnums=(0, ))
def M_fn(u_fn, theta_flat, x):
    '''
    Assemble the M matrix.
    '''
    U_dtheta = jax.vmap(jax.grad(u_fn), (None, 0))
    u_dth = U_dtheta(theta_flat, x)
    M = jnp.mean(u_dth[:, :, jnp.newaxis] * u_dth[:, jnp.newaxis, :], axis=0)
    return M


# @jax.jit
@partial(jax.jit, static_argnums=(0, 1, ))
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
@partial(jax.jit, static_argnums=(0, 1, ))
def r_fn(u_fn, rhs, theta_flat, delta_theta_flat, x, t):
    '''
    Compute the local-in-time residual.
    '''
    x = x.reshape(-1, 1)
    U_dtheta = jax.vmap(jax.grad(u_fn), (None, 0))
    r = jnp.dot(U_dtheta(theta_flat, x), delta_theta_flat) - rhs(theta_flat, x, t, u_fn)
    return r.squeeze()


@partial(jax.jit, static_argnums=(0, 1, ))
def r_loss(u_fn, rhs, theta_flat, delta_theta_flat, x, t):
    '''
    Compute the square norm of the residual.
    '''
    r = r_fn(u_fn, rhs, theta_flat, delta_theta_flat, x, t)
    return jnp.linalg.norm(r) ** 2


@jax.jit
def orthogonalize(u_dth, M_proj):
	'''
	Implementation of the modified Gram-Schmidt orthonormalization algorithm.
	Adapted from: https://github.com/tensorflow/probability/blob/v0.23.0/tensorflow_probability/python/math/gram_schmidt.py#L28
	'''
	n = u_dth.shape[-1]

	# def cond_fn(state):
	# 	_, i = state
	# 	return i < n - 1

	def body_fn(i, vecs):
		# Slice out the vector w.r.t. which we're orthogonalizing the rest.
		vec_norm = jnp.linalg.norm(vecs[:, i])
		u = jnp.divide(vecs[:, i], vec_norm) # (d, )
		# Find weights by dotting the d x 1 against the d x n.
		# weights = jnp.einsum('dm,dn->n', u[:, jnp.newaxis], vecs) # (n, )
        # weights = M_proj[i] # (n, )
		weights = jnp.divide(M_proj[i], jnp.einsum('ii->i', M_proj)) # (n, ) # <u_dth_i, u_dth_j> / <u_dth_i, u_dth_i>
		# Project out vector `u` from the trailing vectors.
		masked_weights = jnp.where(jnp.arange(n) > i, weights, 0.)[jnp.newaxis, :] # (1, n)
		vecs = vecs - jnp.multiply(u[:, jnp.newaxis], masked_weights) # (d, n)
		vecs = jnp.where(jnp.isnan(vecs), 0.0, vecs)
		vecs = jnp.reshape(vecs, u_dth.shape)
		return vecs

	u_dth = jax.lax.fori_loop(0, n, body_fn, u_dth)
	vec_norm = jnp.linalg.norm(u_dth, ord=2, axis=0, keepdims=True)
	u_dth = jnp.divide(u_dth, vec_norm)
	return jnp.where(jnp.isnan(u_dth), 0.0, u_dth)


# @partial(jax.jit, static_argnums=(0, 1, ))
def assemble_weighted(u_fn, rhs, theta_flat, x, t, w_fn):
    '''
    Assemble the weighted M and F matrices.
    '''
    U_dtheta = jax.vmap(jax.grad(u_fn), (None, 0))
    u_dth = U_dtheta(theta_flat, x)
    u_dth_proj = U_dtheta(theta_flat, x)
    M_proj = jnp.mean(u_dth_proj[:, :, jnp.newaxis] * u_dth_proj[:, jnp.newaxis, :], axis=0) # {M_proj}_ij = <u_dth_i, u_dth_j>
    u_dth_ort = orthogonalize(u_dth, M_proj)

    M = jnp.mean(w_fn(x)[:, jnp.newaxis, jnp.newaxis] * (u_dth_ort[:, :, jnp.newaxis] * u_dth_ort[:, jnp.newaxis, :]), axis=0)

    f = rhs(theta_flat, x, t, u_fn) # source term
    F = jnp.mean(w_fn(x)[:, jnp.newaxis] * (u_dth_ort[:, :] * f[:, jnp.newaxis]), axis=0)

    return M, F


# # @jax.jit
# # @partial(jax.jit, static_argnums=(0, ))
# def M_fn_weighted(u_fn, theta_flat, x, w_fn):
#     '''
#     Assemble the weighted M matrix.
#     Note: x must be sampled from the weighted distribution.
#     '''
#     U_dtheta = jax.vmap(jax.grad(u_fn), (None, 0))
#     u_dth = U_dtheta(theta_flat, x)
#     # u_dth_ort, rank = orthogonalize(u_dth)
#     # w = rank / jnp.sum(u_dth_ort ** 2, axis=1)
#     # u_dth_ort = jnp.linalg.qr(u_dth)[0]
#     # w = u_dth_ort.shape[1] / jnp.sum(u_dth_ort ** 2, axis=1)
#     u_dth_proj = U_dtheta(theta_flat, x)
#     M_proj = jnp.mean(u_dth_proj[:, :, jnp.newaxis] * u_dth_proj[:, jnp.newaxis, :], axis=0) # {M_proj}_ij = <u_dth_i, u_dth_j>
#     u_dth_ort = orthogonalize(u_dth, M_proj)
#     # M = jnp.mean(w_fn(x)[:, jnp.newaxis, jnp.newaxis] * (u_dth[:, :, jnp.newaxis] * u_dth[:, jnp.newaxis, :]), axis=0)
#     M = jnp.mean(w_fn(x)[:, jnp.newaxis, jnp.newaxis] * (u_dth_ort[:, :, jnp.newaxis] * u_dth_ort[:, jnp.newaxis, :]), axis=0)
#     return M


# # @jax.jit
# # @partial(jax.jit, static_argnums=(0, 1, ))
# def F_fn_weighted(u_fn, rhs, theta_flat, x, t, w_fn):
#     '''
#     Assemble the weighted F matrix.
#     Note: x must be sampled from the weighted distribution.
#     '''
#     U_dtheta = jax.vmap(jax.grad(u_fn), (None, 0))
#     u_dth = U_dtheta(theta_flat, x)
#     # u_dth_ort, rank = orthogonalize(u_dth)
#     # w = rank / jnp.sum(u_dth_ort ** 2, axis=1)
#     # u_dth_ort = jnp.linalg.qr(u_dth)[0]
#     # w = u_dth_ort.shape[1] / jnp.sum(u_dth_ort ** 2, axis=1)
#     f = rhs(theta_flat, x, t, u_fn) # source term
#     F = jnp.mean(w_fn(x)[:, jnp.newaxis] * (u_dth[:, :] * f[:, jnp.newaxis]), axis=0)
#     return F