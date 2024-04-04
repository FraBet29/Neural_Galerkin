import jax
import jax.numpy as jnp


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


def predictor_corrector(theta_flat, x, t, dt, M_fn, F_fn):
    '''
    Predictor-corrector scheme based on forward Euler.
    '''
    M = M_fn(theta_flat, x)
    F = F_fn(theta_flat, x, t)
    theta_flat_pred = theta_flat + dt * jnp.linalg.lstsq(M, F)[0]
    return theta_flat_pred


def adaptive_sampling(theta_flat, problem_data, n, x, t, M_fn, F_fn, r_fn):
    '''
    Adaptive sampling with SVGD.
    '''

    alpha = 1 # scaling parameter... WHICH VALUE?
    gamma = 0.25 # tempering parameter
    epsilon = 0.05 # step size
    steps = 500

    # Predictor-corrector scheme
    theta_flat_pred = predictor_corrector(theta_flat, x, t, problem_data.dt, M_fn, F_fn)

    # The target measure is proportional to the residual scaled by a tempering parameter
    mu = lambda y: jnp.abs(r_fn(theta_flat, theta_flat_pred - theta_flat, y, t)) ** (2 * gamma)
    log_mu = lambda y: jnp.log(mu(y)) # log(mu) = - V
    log_mu_dx = jax.vmap(jax.grad(log_mu), 0)

    # Gaussian kernel
    sigma = 0.05 # bandwidth
    kernel = lambda x1, x2: jnp.exp(- jnp.abs(x1 - x2) ** 2 / (2 * sigma ** 2))
    kernel_dx = lambda x1, x2: - (x1 - x2) * kernel(x1, x2) / sigma ** 2

    for s in range(steps):
        x_curr = []
        phi = lambda y: alpha / n * jnp.sum(kernel(x, y) * log_mu_dx(x) + kernel_dx(x, y))
        for i in range(n):
            x_curr.append(x[i] + epsilon * phi(x[i]))
        x = jnp.array(x_curr)

    return x