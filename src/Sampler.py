import jax
import jax.numpy as jnp
from functools import partial
import scipy
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


def uniform_sampling(problem_data, n, key=jax.random.key(0)):
    '''
    Uniform sampling in the spatial domain.
    '''
    x = jax.random.uniform(key, (n, problem_data.d), minval=problem_data.domain[0], maxval=problem_data.domain[1])
    return x, jax.random.split(key)[0]


@jax.jit
def SVGD_kernel(z, h):
    '''
    Function adapted from: https://github.com/dilinwang820/Stein-Variational-Gradient-Descent/blob/master/python/svgd.py
    '''
    z_norm_squared = jnp.sum(z ** 2, axis=1, keepdims=True)
    pairwise_dists = z_norm_squared + z_norm_squared.T - 2 * jnp.dot(z, z.T)
    # if h < 0: # median trick
    #     h = jnp.median(pairwise_dists)  
    #     h = jnp.sqrt(0.5 * h / jnp.log(theta.shape[0] + 1))

    Kxy = jnp.exp(- pairwise_dists / h ** 2 / 2) # RBF kernel

    dxkxy = - jnp.matmul(Kxy, z)
    sumkxy = jnp.sum(Kxy, axis=1)
    dxkxy += jnp.multiply(z, jnp.expand_dims(sumkxy, axis=1))
    dxkxy /= (h ** 2)
    return (Kxy, dxkxy)


def SVGD_update(z0, log_mu_dx, steps=1000, epsilon=1e-3, alpha=1.0, diagnostic_on=False):
    '''
    Function adapted from: https://github.com/dilinwang820/Stein-Variational-Gradient-Descent/blob/master/python/svgd.py
    '''
    z = jnp.copy(z0)

    if diagnostic_on:
        z_old = jnp.copy(z)
        wass = []

    for s in range(steps):

        log_mu_dx_val = log_mu_dx(z.squeeze()).reshape(-1, 1) # log_mu_dx: (n, d)
        # Calculating the kernel matrix
        kxy, dxkxy = SVGD_kernel(z, h=0.05) # kxy: (n, n), dxkxy: (n, d)
        
        # # Print some info about the kernel
        # print('det', jnp.linalg.det(kxy))
        # print('min eig', jnp.min(jnp.linalg.eigvals(kxy)))
        # print('max eig', jnp.max(jnp.linalg.eigvals(kxy)))
        
        grad_z = alpha * (jnp.matmul(kxy, log_mu_dx_val) + dxkxy) / z0.shape[0] # grad_x: (n, d)
        
        # Vanilla update
        z = z + epsilon * grad_z
        # epsilon = epsilon * 0.9 # annealing
        # epsilon = epsilon / jnp.sqrt(s + 1) # Robbins-Monro

        # adagrad with momentum
        # fudge_factor = 1e-6
        # historical_grad = 0
        # beta = 0.9

        # if s == 0:
        #     historical_grad = historical_grad + grad_z ** 2
        # else:
        #     historical_grad = beta * historical_grad + (1 - beta) * (grad_z ** 2)
        # adj_grad = jnp.divide(grad_z, fudge_factor + jnp.sqrt(historical_grad))

        # z = z + epsilon * adj_grad

        if diagnostic_on:
            wass.append(wasserstein_1d(z_old.squeeze(), z.squeeze(), p=2))
            z_old = jnp.copy(z)
    
    if diagnostic_on:
        # Plot Wasserstein distance (cumsum)
        plt.plot(jnp.cumsum(jnp.array(wass)) / (jnp.arange(s + 1) + 1))
        plt.title('Wasserstein distance (cumsum)')
        plt.xlabel('SVGD iters')
        plt.ylabel('Wasserstein distance (cumsum)')
        plt.show()
        # Plot Wasserstein distance (movmean)
        def movmean(x, w):
            return jnp.convolve(x, jnp.ones(w), 'valid') / w
        plt.plot(movmean(jnp.array(wass), 100))
        plt.title('Wasserstein distance (movmean)')
        plt.xlabel('SVGD iters')
        plt.ylabel('Wasserstein distance (movmean)')
        plt.show()
    
    # print(f'SVGD iterations: {s + 1}')

    return z


def SVGD_update_adaptive(z0, log_mu_dx, steps=100, epsilon=1e-1, alpha=1.0, diagnostic_on=False):
    '''
    Function adapted from: https://github.com/dilinwang820/Stein-Variational-Gradient-Descent/blob/master/python/svgd.py
    '''
    z = jnp.copy(z0) # (n, d)

    if diagnostic_on:
        z_old = jnp.copy(z)
        wass = []

    def rhs_svgd(t, z):
        log_mu_dx_val = log_mu_dx(z.squeeze()).reshape(-1, 1) # log_mu_dx: (n, d)
        kxy, dxkxy = SVGD_kernel(z.reshape(-1, 1), h=0.05) # kxy: (n, n), dxkxy: (n, d)
        grad_z = alpha * (jnp.matmul(kxy, log_mu_dx_val) + dxkxy) / z0.shape[0] # grad_x: (n, d)
        return grad_z.squeeze()

    # scheme = scipy.integrate.RK45(rhs_svgd, 0, z.squeeze(), steps, max_step=epsilon, rtol=1e-4)
    scheme = scipy.integrate.RK45(rhs_svgd, 0, z.squeeze(), 1, rtol=1e-4)

    # for s in range(steps):
    while scheme.t < 1:
        
        scheme.step()
        z = scheme.y

        if diagnostic_on:
            wass.append(wasserstein_1d(z_old.squeeze(), z.squeeze(), p=2))
            z_old = jnp.copy(z)
    
    if diagnostic_on:
        # plt.plot(jnp.cumsum(jnp.array(wass)) / (jnp.arange(s + 1) + 1))
        plt.plot(jnp.cumsum(jnp.array(wass)) / (jnp.arange(len(wass)) + 1))
        plt.title('Wasserstein distance (cumsum)')
        plt.xlabel('SVGD iters')
        plt.ylabel('Wasserstein distance (cumsum)')
        plt.show()

    return z.reshape(-1, 1)


def SVGD_update_corrected(z0, log_mu_dx, steps=1000, epsilon=1e-3, alpha=1.0, diagnostic_on=False):
    '''
    Function adapted from: https://github.com/dilinwang820/Stein-Variational-Gradient-Descent/blob/master/python/svgd.py
    '''
    N, d = z0.shape
    key = jax.random.key(0)

    z = jnp.copy(z0)

    if diagnostic_on:
        z_old = jnp.copy(z)
        wass = []

    for s in range(steps):

        log_mu_dx_val = log_mu_dx(z.squeeze()).reshape(-1, 1) # log_mu_dx: (n, d)
        # Calculating the kernel matrix
        kxy, dxkxy = SVGD_kernel(z, h=0.05) # kxy: (n, n), dxkxy: (n, d)
        grad_z = alpha * (jnp.matmul(kxy, log_mu_dx_val) + dxkxy) / z0.shape[0]
        # print('Min eig of kxy (corrected):', jnp.min(jnp.linalg.eigvals(kxy + 1e-3 * jnp.eye(N))))
        # L = jnp.linalg.cholesky(kxy + 1e-3 * jnp.eye(N)) # correction needed for numerical stability
        # z = z + epsilon * grad_z + jnp.sqrt(2 * epsilon / z0.shape[0]) * L @ jax.random.normal(key, (N, d))
        U, S, _ = jnp.linalg.svd(kxy) # more robust than Cholesky, but slower
        z = z + epsilon * grad_z + jnp.sqrt(2 * epsilon / z0.shape[0]) * U @ jnp.diag(jnp.sqrt(S)) @ jax.random.normal(key, (N, d))
        # epsilon = epsilon * 0.9 # annealing
        key = jax.random.split(key)[0]

        if diagnostic_on:
            wass.append(wasserstein_1d(z_old.squeeze(), z.squeeze(), p=2))
            z_old = jnp.copy(z)
    
    if diagnostic_on:
        plt.plot(jnp.cumsum(jnp.array(wass)) / (jnp.arange(s + 1) + 1))
        plt.title('Wasserstein distance (cumsum)')
        plt.xlabel('SVGD iters')
        plt.ylabel('Wasserstein distance (cumsum)')
        plt.show()

    return z


def high_order_runge_kutta(x0, log_mu_dx, T=100, h=0.05, alpha=1.0, diagnostic_on=False):
    
    x0 = x0.squeeze()
    x = jnp.copy(x0)

    if diagnostic_on:
        x_old = jnp.copy(x)
        wass = []
	
    sigma = jnp.sqrt(2 * alpha)
    f = lambda x: alpha * log_mu_dx(x) # f = - alpha * grad(V) = alpha * grad(log(mu))

    def runge_kutta_step(x, t, h, key):
        csi = jax.random.normal(key, (x.shape[0], ))
        y1 = x + jnp.sqrt(2 * h) * sigma * csi
        y2 = x - 3/8 * h * f(y1) + jnp.sqrt(2 * h) * sigma * csi / 4
        x = x - 1/3 * h * f(y1) + 4/3 * h * f(y2) + sigma * jnp.sqrt(h) * csi
        return x, jax.random.split(key)[0]
    
    # def runge_kutta_step(x, t, h, key):
    #     csi = jax.random.normal(key, (x.shape[0],))
    #     y1 = x + sigma * jnp.sqrt(h) * csi
    #     y2 = x - h/2 * f(y1) + sigma/2 * jnp.sqrt(h) * csi
    #     y3 = x + 3 * h * f(y1) - 2 * h * f(y2) + sigma * jnp.sqrt(h) * csi
    #     x = x - 3/2 * h * f(y1) + 2 * h * f(y2) + h/2 * f(y3) + sigma * jnp.sqrt(h) * csi
    #     return x, jax.random.split(key)[0]

    key = jax.random.key(0)

    for t in range(T):
        x, key = runge_kutta_step(x, t, h, key)
        # h = h / jnp.sqrt(t + 1) # Robbins-Monro
        # h = 0.9 * h # annealing
        if diagnostic_on:
            wass.append(wasserstein_1d(x_old.squeeze(), x.squeeze(), p=2))
            x_old = jnp.copy(x)
    
    if diagnostic_on:
        plt.plot(jnp.cumsum(jnp.array(wass)) / (jnp.arange(len(wass)) + 1))
        plt.title('Wasserstein distance (cumsum)')
        plt.xlabel('SVGD iters')
        plt.ylabel('Wasserstein distance (cumsum)')
        plt.show()

    return x.reshape(-1, 1)


def high_order_sampler(u_fn, rhs, theta_flat, problem_data, x, t, gamma=1.0, h=0.01, steps=500, diagnostic_on=False):
    '''
    Adaptive sampling with high-order Runge-Kutta-like method.
    '''
    # Define the scaling parameter
    # alpha = problem_data.dt / h
    alpha = 1e-2 * min(problem_data.dt / h, 1.0)

    # Predictor-corrector scheme
    delta_theta_flat = predictor_corrector(u_fn, rhs, theta_flat, x, t)

    # The target measure is proportional to the residual scaled by a tempering parameter
    mu = lambda y: jnp.abs(r_fn(u_fn, rhs, theta_flat, delta_theta_flat, y, t)) ** (2 * gamma) # sample from residual
    log_mu = lambda y: jnp.log(mu(y)) # log(mu) = - V
    log_mu_dx = jax.vmap(jax.grad(log_mu), 0)

    x = high_order_runge_kutta(x, log_mu_dx, T=steps, h=h, alpha=alpha, diagnostic_on=diagnostic_on)
    
    # Constrain the particles in the spatial domain
    x = jnp.clip(x, problem_data.domain[0], problem_data.domain[1])

    return x


# @jax.jit
# def K(x, h=0.05):
#     xmx = jnp.expand_dims(x, 0) - jnp.expand_dims(x, 1)
#     norm = jnp.einsum('ijk,ijk->ij', xmx, xmx)
#     Kval = jnp.exp(- norm / h ** 2 / 2)
#     gKval = jnp.expand_dims(Kval, -1) * (xmx / h ** 2)
#     return (Kval, gKval)


# def svgd(x0, g_logp, T=1000, eta=1e-3, alpha=1.0):

#     @jax.jit
#     def update_svgd(x):
#         Kval, gKval = K(x)
#         return alpha * (Kval @ g_logp(x.squeeze()).reshape(-1, 1) + gKval.sum(0)) / x0.shape[0]
    
#     def body_fn(i, x):
#         return x + eta * update_svgd(x) 

#     x = jax.lax.fori_loop(0, T, body_fn, x0)

#     return x


# def svgd_unbiased(x0, g_logp, T=1000, eta=1e-3, alpha=1.0, key=jax.random.key(0)):

#     N, d = x0.shape

#     @jax.jit
#     def update_svgd(x, key):
#         Kval, gKval = K(x)
#         # U, D, V = jnp.linalg.svd(Kval)
#         # return (eta * (Kval @ g_logp(x.squeeze()).reshape(-1, 1) + gKval.sum(0)) + (U @ jnp.diag(jnp.sqrt(2 * eta * D)) @ V) @ jax.random.normal(key, (N, d)), 
#         #        jax.random.split(key)[0])
#         L = jnp.linalg.cholesky(2 * eta * Kval + 1e-6 * jnp.eye(N)) # correction needed for numerical stability
#         return (eta * alpha * (Kval @ g_logp(x.squeeze()).reshape(-1, 1) + gKval.sum(0)) / x0.shape[0] + L @ jax.random.normal(key, (N, d)), 
#                 jax.random.split(key)[0])
    
#     def body_fn(i, state):
#         x, key = state
#         x, key = update_svgd(x, key)
#         return (x, key)
    
#     x, _ = jax.lax.fori_loop(0, T, body_fn, (x0, key))
    
#     return x


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


def adaptive_sampling(u_fn, rhs, theta_flat, problem_data, x, t, gamma=1.0, epsilon=0.01, steps=500, corrected=False, diagnostic_on=False):
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

    if corrected:
        x = SVGD_update_corrected(x, log_mu_dx, steps, epsilon, alpha, diagnostic_on=diagnostic_on)
    else:
        x = SVGD_update(x, log_mu_dx, steps, epsilon, alpha, diagnostic_on=diagnostic_on)
        # x = SVGD_update_adaptive(x, log_mu_dx, alpha=alpha, diagnostic_on=diagnostic_on)
        # x = high_order_runge_kutta(x, mu, alpha=alpha, diagnostic_on=diagnostic_on)
    
    # x = svgd(x, log_mu_dx, T=steps, eta=epsilon)
    # x = svgd_unbiased(x, log_mu_dx, T=steps, eta=epsilon)

	# Constrain the particles in the spatial domain
    x = jnp.clip(x, problem_data.domain[0], problem_data.domain[1])

    return x


# Class for storing the orthonormal basis and the Gram-Schmidt change of basis
class Orth():

	def __init__(self, Q=None, B_GS=None):
		self.Q = Q
		self.B_GS = B_GS
 
	def __call__(self, x):
		return self.Q, self.B_GS

	def reset(self):
		self.Q = None
		self.B_GS = None


@partial(jax.jit, static_argnums=(0, ))
def orthogonalize(u_dth_fun, theta_flat, x_proj, L):
	'''
	Implementation of the modified Gram-Schmidt orthonormalization algorithm.
	Adapted from: https://github.com/tensorflow/probability/blob/v0.23.0/tensorflow_probability/python/math/gram_schmidt.py#L28
	'''
	# print('Orthogonalization...')

	# u_dth = u_dth_fun(x)
	u_dth_proj = u_dth_fun(theta_flat, x_proj.reshape(-1, 1)) # (n_proj, p)
	p = u_dth_proj.shape[-1]

	def body_fn(i, vecs): # the initial vectors in vecs are progressively replaced by the orthonormalized ones
		for _ in range(2): # reorthogonalization
			vec_norm = jnp.sqrt(jnp.mean(vecs[:, i] ** 2) * L)
			u = jnp.divide(vecs[:, i], vec_norm) # (n_proj, )
			weights = jnp.mean(u[:, jnp.newaxis] * vecs, axis=0) * L # (p, )
			# weights = jnp.where(weights > 1e-6 * jnp.max(weights), weights, 0.0) # remove the basis vectors with zero norm
			# vecs = jnp.where(weights > 1e-6 * jnp.max(weights), vecs, 0.0) # remove the basis vectors with zero norm
			masked_weights = jnp.where(jnp.arange(p) > i, weights, 0.)[jnp.newaxis, :] # (1, p) # consider only the first i vectors
			vecs = vecs - jnp.multiply(u[:, jnp.newaxis], masked_weights) # (n_proj, p)
			vecs = jnp.where(jnp.isnan(vecs), 0.0, vecs)
			vecs = jnp.reshape(vecs, u_dth_proj.shape)
		return vecs

	u_dth_proj = jax.lax.fori_loop(0, p, body_fn, u_dth_proj)
	vec_norm = jnp.sqrt(jnp.mean(u_dth_proj * u_dth_proj, axis=0) * L)
	# print('Norms of the orthonormal basis:', vec_norm)
	# u_dth_proj = jnp.where(vec_norm > 1e-6 * jnp.max(vec_norm), u_dth_proj, 0.0) # remove the basis vectors with zero norm
	u_dth_proj = jnp.divide(u_dth_proj, vec_norm)
	u_dth_proj = jnp.where(jnp.isnan(u_dth_proj), 0.0, u_dth_proj)
	# vec_norm = jnp.where(vec_norm > 1e-6 * jnp.max(vec_norm), vec_norm, 0.0) # remove the basis vectors with zero norm

	# orthonormality check
	# for i in range(p):
	# 	for j in range(p):
	# 		inner_prod = jnp.mean(u_dth_proj[:, i] * u_dth_proj[:, j]) * L
	# 		if i != j and inner_prod > 1e-3:
	# 			print(f'### Warning: inner product ({i}, {j}) = {inner_prod} ###')

	# recover the gram-schmidt change of basis matrix

	B_GS_diag = vec_norm.squeeze()
	# B_GS_off = jnp.mean(u_dth_fun(x_proj)[:, :, jnp.newaxis] * u_dth_proj[:, jnp.newaxis, :], axis=0) * L
	B_GS_off = jnp.mean(u_dth_proj[:, :, jnp.newaxis] * u_dth_fun(theta_flat, x_proj.reshape(-1, 1))[:, jnp.newaxis, :], axis=0) * L
	B_GS = jnp.diag(B_GS_diag) + jnp.triu(B_GS_off, k=1).T

	return u_dth_proj, B_GS


def weighted_sampling(u_fn, theta_flat, problem_data, x, store, gamma=1.0, epsilon=0.01, steps=100):
    '''
    Weighted sampling in the spatial domain.
    '''
    # print('Weighted sampling...')

    # Define the scaling parameter
    alpha = problem_data.dt / epsilon

    def w_fn(x, store): # returns the weights, the orthonormal basis, and the Gram-Schmidt basis

        # function returning the gradient components
        U_dtheta = jax.vmap(jax.grad(u_fn), (None, 0))
        p = U_dtheta(theta_flat, x.reshape(-1, 1)).shape[1] # number of parameters
        
        # compute orthonormal basis evaluated on x_proj
        L = problem_data.domain[1] - problem_data.domain[0]
        x_proj = jax.random.uniform(jax.random.key(0), (problem_data.N, )) * L
        x_proj = jnp.sort(x_proj) # needed for interpolation
        
        if store.Q is None:
            Q_proj, B_GS = orthogonalize(U_dtheta, theta_flat, x_proj, L)
            store.Q = Q_proj
            store.B_GS = B_GS
            # print('Condition number of the Gram-Schmidt basis:', jnp.linalg.cond(B_GS))
        else:
            Q_proj, B_GS = store.Q, store.B_GS

        if x.ndim == 0:
            n = 1
        else:
            n = x.shape[0]

        # evaluate the orthonormal basis on x via interpolation
        Q = jnp.array([jnp.interp(x, x_proj, q).reshape(-1) for q in Q_proj.T]).T # (n, p)
        # Q = jnp.zeros((n, p))
        # for i in range(p):
        #     Q = Q.at[:, i].set(jnp.interp(x, x_proj, Q_proj[:, i]))

        return (p / jnp.sum(Q ** 2, axis=1)).squeeze(), Q, B_GS # (n, )
    
    # When called for SVGD: one sample at a time
    # When called for weighted sampling: multiple samples at a time
    
    # The target measure is nu(x) / w(x) = 1 / w(x) in the uniform case
    mu = lambda y: 1 / w_fn(y, store)[0]
    log_mu = lambda y: jnp.log(mu(y)) # log(mu) = - V
    log_mu_dx = jax.vmap(jax.grad(log_mu), 0)

    x = SVGD_update(x, log_mu_dx, steps, epsilon, alpha)
    
    # Constrain the particles in the spatial domain
    x = jnp.clip(x, problem_data.domain[0], problem_data.domain[1])

    return x, w_fn, store