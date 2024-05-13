import jax
import jax.numpy as jnp
from functools import partial
import scipy
from scipy.stats import uniform
from Assemble import *
from Diagnostic import *


# def sampler(sample_mode, *args):
#     '''
#     Sampler function.
#     '''
#     if sample_mode == 'uniform':
#         return uniform_sampling(*args)
#     elif sample_mode == 'adaptive':
#         return adaptive_sampling(*args)
#     else:
#         raise ValueError(f'Unknown sample mode: {sample_mode}.')


def uniform_sampling(problem_data, n, seed=0):
    '''
    Uniform sampling in the spatial domain.
    '''
    x = jax.random.uniform(jax.random.key(seed), (n, problem_data.d), minval=problem_data.domain[0], maxval=problem_data.domain[1])
    return x


@jax.jit
def SVGD_kernel(z, h):
    '''
    Function adapted from: https://github.com/dilinwang820/Stein-Variational-Gradient-Descent/blob/master/python/svgd.py
    '''
    # sq_dist = pdist(theta)
    # pairwise_dists = squareform(sq_dist) ** 2
    z_norm_squared = jnp.sum(z ** 2, axis=1, keepdims=True)
    pairwise_dists = z_norm_squared + z_norm_squared.T - 2 * jnp.dot(z, z.T)
    # if h < 0: # median trick
    #     h = jnp.median(pairwise_dists)  
    #     h = jnp.sqrt(0.5 * h / jnp.log(theta.shape[0] + 1))

    # compute the rbf kernel
    Kxy = jnp.exp(- pairwise_dists / h ** 2 / 2)

    dxkxy = - jnp.matmul(Kxy, z)
    sumkxy = jnp.sum(Kxy, axis=1)
    # for i in range(z.shape[1]):
    #     dxkxy = dxkxy.at[:, i].set(dxkxy[:, i] + jnp.multiply(z[:, i], sumkxy))
    dxkxy += jnp.multiply(z, jnp.expand_dims(sumkxy, axis=1)) # vectorized
    dxkxy /= (h ** 2)
    return (Kxy, dxkxy)


def SVGD_update(z0, log_mu_dx, steps=1000, epsilon=1e-3, alpha=1.0, diagnostic_on=False):
    '''
    Function adapted from: https://github.com/dilinwang820/Stein-Variational-Gradient-Descent/blob/master/python/svgd.py
    '''

    ### NEW VERSION ###

    def body_fn(i, z):
        log_mu_dx_val = log_mu_dx(z.squeeze()).reshape(-1, 1)
        kxy, dxkxy = SVGD_kernel(z, h=0.05)
        grad_z = alpha * (jnp.matmul(kxy, log_mu_dx_val) + dxkxy) / z0.shape[0]
        z = z + epsilon * grad_z
        return z

    z = jax.lax.fori_loop(0, steps, body_fn, z0)

    return z

    ### OLD VERSION ###

    # z = jnp.copy(z0)

    # if diagnostic_on:
    #     z_old = jnp.copy(z)
    #     wass = []

    # for s in range(steps):

    #     log_mu_dx_val = log_mu_dx(z.squeeze()).reshape(-1, 1) # log_mu_dx: (n, d)
    #     # Calculating the kernel matrix
    #     kxy, dxkxy = SVGD_kernel(z, h=0.05) # kxy: (n, n), dxkxy: (n, d)
    #     grad_z = alpha * (jnp.matmul(kxy, log_mu_dx_val) + dxkxy) / z0.shape[0] # grad_x: (n, d)
    #     # Vanilla update
    #     z = z + epsilon * grad_z

    #     if diagnostic_on:
    #         wass.append(wasserstein_1d(z_old.squeeze(), z.squeeze(), p=2))
    #         z_old = jnp.copy(z)
    
    # if diagnostic_on:
    #     plt.plot(jnp.cumsum(jnp.array(wass)) / (jnp.arange(s + 1) + 1))
    #     plt.title('Wasserstein distance (cumsum)')
    #     plt.xlabel('SVGD iters')
    #     plt.ylabel('Wasserstein distance (cumsum)')
    #     plt.show()
    
    # # print(f'SVGD iterations: {s + 1}')

    # return z


@jax.jit
def K(x, h=0.05):
    xmx = jnp.expand_dims(x, 0) - jnp.expand_dims(x, 1)
    norm = jnp.einsum('ijk,ijk->ij', xmx, xmx)
    Kval = jnp.exp(- norm / h ** 2 / 2)
    gKval = jnp.expand_dims(Kval, -1) * (xmx / h ** 2)
    return (Kval, gKval)


def svgd(x0, g_logp, T=1000, eta=1e-3, alpha=1.0):

    @jax.jit
    def update_svgd(x):
        Kval, gKval = K(x)
        return alpha * (Kval @ g_logp(x.squeeze()).reshape(-1, 1) + gKval.sum(0)) / x0.shape[0]
    
    def body_fn(i, x):
        return x + eta * update_svgd(x) 

    x = jax.lax.fori_loop(0, T, body_fn, x0)

    return x


def svgd_unbiased(x0, g_logp, T=1000, eta=1e-3, key=jax.random.PRNGKey(0)):

    N, d = x0.shape

    @jax.jit
    def update_svgd(x, key):
        Kval, gKval = K(x)
        # U, D, V = jnp.linalg.svd(Kval)
        # return (eta * (Kval @ g_logp(x.squeeze()).reshape(-1, 1) + gKval.sum(0)) + (U @ jnp.diag(jnp.sqrt(2 * eta * D)) @ V) @ jax.random.normal(key, (N, d)), 
        #        jax.random.split(key)[0])
        L = jnp.linalg.cholesky(2 * eta * Kval + 1e-6 * jnp.eye(N)) # correction needed for numerical stability
        return (eta * (Kval @ g_logp(x.squeeze()).reshape(-1, 1) + gKval.sum(0)) + L @ jax.random.normal(key, (N, d)), 
                jax.random.split(key)[0])
    
    def body_fn(i, state):
        x, key = state
        x, key = update_svgd(x, key)
        return (x, key)
    
    x, _ = jax.lax.fori_loop(0, T, body_fn, (x0, key))
    
    return x


# NOTE: need to compute theta_flat or delta_theta_flat?
@partial(jax.jit, static_argnums=(0,1,))
def predictor_corrector(u_fn, rhs, theta_flat, x, t):
    '''
    Predictor-corrector scheme based on forward Euler.
    '''
    return jnp.linalg.lstsq(M_fn(u_fn, theta_flat, x), F_fn(u_fn, rhs, theta_flat, x, t))[0]
    # M = M_fn(u_fn, theta_flat, x)
    # F = F_fn(u_fn, rhs, theta_flat, x, t)
    # theta_flat_pred = theta_flat + dt * jnp.linalg.lstsq(M, F)[0]
    # return theta_flat_pred


def adaptive_sampling(u_fn, rhs, theta_flat, problem_data, x, t, gamma=1.0, epsilon=0.01, steps=500, diagnostic_on=False):
    '''
    Adaptive sampling with SVGD.
    Args:
        u_fn: neural network
        rhs: source term
        theta_flat: flattened parameters
        problem_data: problem data
        x: initial points
        t: current time
        epsilon: step size
        gamma: tempering parameter
        steps: number of iterations
    '''
    # Define the scaling parameter
    alpha = problem_data.dt / epsilon
    # alpha related to the Wiener process in the Langevin dynamics

    # Predictor-corrector scheme
    delta_theta_flat = predictor_corrector(u_fn, rhs, theta_flat, x, t)

    # The target measure is proportional to the residual scaled by a tempering parameter
    # mu = lambda y: jnp.abs(r_fn(theta_flat, theta_flat_pred - theta_flat, y, t)) ** (2 * gamma)
    mu = lambda y: jnp.abs(r_fn(u_fn, rhs, theta_flat, delta_theta_flat, y, t)) ** (2 * gamma) # sample from residual
    # U = jax.vmap(u_fn, (None, 0))
    # mu = lambda y: jnp.abs(U(theta_flat, y.reshape(-1, 1))).squeeze() ** gamma # sample from solution
    log_mu = lambda y: jnp.log(mu(y)) # log(mu) = - V
    log_mu_dx = jax.vmap(jax.grad(log_mu), 0)

    x = SVGD_update(x, log_mu_dx, steps, epsilon, alpha, diagnostic_on=diagnostic_on)
    # x = svgd(x, log_mu_dx, T=steps, eta=epsilon)
    # x = svgd_unbiased(x, log_mu_dx, T=steps, eta=epsilon)

	# Constrain the particles in the spatial domain
    x = jnp.clip(x, problem_data.domain[0], problem_data.domain[1])

    return x


# @partial(jax.jit, static_argnums=(0, ))
# def orthogonalize(u_fn, theta_flat, problem_data, u_dth, eps=1e-15):
    
#     # jax.config.update('jax_enable_x64', True) # enable double precision
#     n = u_dth.shape[1] # number of parameters
#     v_dth = u_dth.T # (p, n)
#     v_dth = v_dth.at[0].divide(jnp.linalg.norm(v_dth[0])) # normalize the first vector

#     # The projection coefficients are w.r.t. the L2 norm
#     U_dtheta = jax.vmap(jax.grad(u_fn), (None, 0))
#     x = jax.random.uniform(jax.random.PRNGKey(0), (100, problem_data.d), 
#                            minval=problem_data.domain[0], maxval=problem_data.domain[1])
#     u_dth_coeff = U_dtheta(theta_flat, x) # (100, p)
#     M_proj = jnp.mean(u_dth_coeff[:, :, jnp.newaxis] * u_dth_coeff[:, jnp.newaxis, :], axis=0) / jnp.mean(u_dth_coeff ** 2, axis=0) # (p, p)
#     # 1st row: proj_11, proj_12, ..., proj_1p
#     # 2nd row: proj_21, proj_22, ..., proj_2p
#     # etc.

#     for i in range(1, n):
#         v_dth_prev = v_dth[0:i] # (i, n)
#         proj = M_proj[0:i, i] # (i, )
#         # V = V.at[i].add(-jnp.dot(proj, V_prev).T)
#         v_dth = v_dth.at[i].add(-jnp.dot(proj, v_dth_prev).T)
#         v_dth = v_dth.at[i].divide(jnp.linalg.norm(v_dth[i]))
#         # if jnp.linalg.norm(v_dth[i]) < eps:
#         #     # V = V.at[i][V[i] < eps].set(0.0)
#         #     v_dth = v_dth.at[i].set(0.0)
#         # else:
#         #     v_dth = v_dth.at[i].divide(jnp.linalg.norm(v_dth[i]))
#     # Compute rank
#     # rank = jnp.sum(jnp.linalg.norm(V, axis=1) > eps)
#     return v_dth.T


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


def weighted_sampling(u_fn, theta_flat, problem_data, x, gamma=1.0, epsilon=0.01, steps=500):
    '''
    Weighted sampling in the spatial domain.
    '''
    # Define the scaling parameter
    alpha = problem_data.dt / epsilon

    def ort_basis(y): # orthonormal basis of the tangent space
        
        y = y.reshape(-1, 1)
        U_dtheta = jax.vmap(jax.grad(u_fn), (None, 0))
        u_dth = U_dtheta(theta_flat, y)

        # u_dth_ort = jnp.linalg.qr(u_dth)[0] # https://github.com/google/jax/blob/main/jax/_src/lax/linalg.py
        # print(u_dth_ort.T @ u_dth_ort)

        x = jax.random.uniform(jax.random.PRNGKey(0), (2048, problem_data.d), 
                                minval=problem_data.domain[0], maxval=problem_data.domain[1])
        u_dth_proj = U_dtheta(theta_flat, x) # (2048, n)
        M_proj = jnp.mean(u_dth_proj[:, :, jnp.newaxis] * u_dth_proj[:, jnp.newaxis, :], axis=0) # {M_proj}_ij = <u_dth_i, u_dth_j>
        # The projection coefficients are w.r.t. the L2 norm (independent of the samples)
        # If some of the vectors are linearly dependent, then the corresponding rows of M_proj will be zero (or very small)
        # cutoff = 1e-3 * jnp.max(jnp.abs(M_proj))
        # idxs = jnp.unique(jnp.where(jnp.abs(M_proj) > cutoff)[1]) # indices of the vectors to keep
        # M_proj = jnp.where(jnp.abs(M_proj) > cutoff, M_proj, 0.0)

        u_dth_ort = orthogonalize(u_dth, M_proj)
        # u_dth_ort = u_dth_ort[:, idxs]

        return u_dth_ort
    
    def w_fn(y):
        u_dth_ort = ort_basis(y)
        return u_dth_ort.shape[1] / jnp.sum(u_dth_ort ** 2, axis=1).squeeze()
    # NOTE: we are assuming here that the dimension of the subspace is the same as the dimension of theta (!)
    # w_fn = lambda y: theta_flat.shape[0] / jnp.sum(ort_basis(y) ** 2, axis=1).squeeze() # optimal weights
    # When called for SVGD: one sample at a time
    # When called for weighted sampling: multiple samples at a time
    
    # The target measure is nu(x) / w(x) = 1 / w(x) in the uniform case
    mu = lambda y: 1 / w_fn(y)
    log_mu = lambda y: jnp.log(mu(y)) # log(mu) = - V
    log_mu_dx = jax.vmap(jax.grad(log_mu), 0)

    x = SVGD_update(x, log_mu_dx, steps, epsilon, alpha)

    return x, w_fn