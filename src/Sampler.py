import jax
import jax.numpy as jnp
from functools import partial
from scipy.stats import uniform
from Assemble import *


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


def SVGD_update(z0, log_mu_dx, steps=1000, epsilon=1e-3, alpha=1.0):
    '''
    Function adapted from: https://github.com/dilinwang820/Stein-Variational-Gradient-Descent/blob/master/python/svgd.py
    '''
    z = jnp.copy(z0)

    for s in range(steps):
        log_mu_dx_val = log_mu_dx(z.squeeze()).reshape(-1, 1) # log_mu_dx: (n, d)
        # Calculating the kernel matrix
        kxy, dxkxy = SVGD_kernel(z, h=0.05) # kxy: (n, n), dxkxy: (n, d)
        grad_z = alpha * (jnp.matmul(kxy, log_mu_dx_val) + dxkxy) / z0.shape[0] # grad_x: (n, d)
        # Vanilla update
        z = z + epsilon * grad_z
        
    return z


# TODO: need to compute theta_flat or delta_theta_flat?
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


def adaptive_sampling(u_fn, rhs, theta_flat, problem_data, x, t, gamma=1.0, epsilon=0.01, steps=500):
    '''
    Adaptive sampling with SVGD.
    Args:
        u_fn: neural network
        rhs: source term
        theta_flat: flattened parameters
        problem_data: problem data
        n: number of points
        x: initial points
        t: current time
        epsilon: step size
        gamma: tempering parameter
        steps: number of iterations
    '''
    # Define the scaling parameter
    alpha = problem_data.dt / epsilon

    # Predictor-corrector scheme
    delta_theta_flat = predictor_corrector(u_fn, rhs, theta_flat, x, t)

    # The target measure is proportional to the residual scaled by a tempering parameter
    # mu = lambda y: jnp.abs(r_fn(theta_flat, theta_flat_pred - theta_flat, y, t)) ** (2 * gamma)
    mu = lambda y: jnp.abs(r_fn(u_fn, rhs, theta_flat, delta_theta_flat, y, t)) ** (2 * gamma)
    log_mu = lambda y: jnp.log(mu(y)) # log(mu) = - V
    log_mu_dx = jax.vmap(jax.grad(log_mu), 0)

    x = SVGD_update(x, log_mu_dx, steps, epsilon, alpha)

    return x