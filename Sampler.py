import jax
import jax.numpy as jnp
from scipy.stats import uniform


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


def sample_from_function(f, g, C, n, max_it=1000):
    '''
    Accept-reject sampling from a function.
    Args:
        f: target function
        g: proposal distribution
        C: constant such that f(x) <= C * g(x) for all x
        n: number of samples
        max_it: maximum number of iterations
    '''
    x = []

    for _ in range(n):
        it = 0
        while it < max_it:
            y = g.rvs() # sample from the Gaussian
            u = uniform.rvs() # sample from a uniform in [0, 1]
            if u <= f(y) / (C * g.pdf(y)):
                x.append(y)
                break
            it += 1
        if it >= max_it:
            raise Exception("AR did not converge.")

    return jnp.array(x)


@jax.jit
def SVGD_kernel(theta, h):
    '''
    Function adapted from: https://github.com/dilinwang820/Stein-Variational-Gradient-Descent/blob/master/python/svgd.py
    '''
    # sq_dist = pdist(theta)
    # pairwise_dists = squareform(sq_dist) ** 2
    theta_norm_squared = jnp.sum(theta ** 2, axis=1, keepdims=True)
    pairwise_dists = theta_norm_squared + theta_norm_squared.T - 2 * jnp.dot(theta, theta.T)
    # if h < 0: # median trick
    #     h = jnp.median(pairwise_dists)  
    #     h = jnp.sqrt(0.5 * h / jnp.log(theta.shape[0] + 1))

    # compute the rbf kernel
    Kxy = jnp.exp(- pairwise_dists / h ** 2 / 2)

    dxkxy = - jnp.matmul(Kxy, theta)
    sumkxy = jnp.sum(Kxy, axis=1)
    # for i in range(theta.shape[1]):
    #     dxkxy = dxkxy.at[:, i].set(dxkxy[:, i] + jnp.multiply(theta[:, i], sumkxy))
    dxkxy += jnp.multiply(theta, jnp.expand_dims(sumkxy, axis=1)) # vectorized
    dxkxy /= (h ** 2)
    return (Kxy, dxkxy)


def SVGD_update(x0, lnprob, n_iter=1000, stepsize=1e-3, alpha=1.0, debug=False):
    '''
    Function adapted from: https://github.com/dilinwang820/Stein-Variational-Gradient-Descent/blob/master/python/svgd.py
    '''    
    theta = jnp.copy(x0) 

    for iter in range(n_iter):
        if debug and (iter + 1) % 1000 == 0:
            print('iter', str(iter + 1))

        lnpgrad = lnprob(theta.squeeze()).reshape(-1, 1) # lnpgrad: (n, d)
        # calculating the kernel matrix
        kxy, dxkxy = SVGD_kernel(theta, h=0.05) # kxy: (n, n), dxkxy: (n, d)
        grad_theta = alpha * (jnp.matmul(kxy, lnpgrad) + dxkxy) / x0.shape[0] # grad_theta: (n, d)
        
        # vanilla update
        theta = theta + stepsize * grad_theta
        
    return theta


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
    alpha = 1.0 # scaling parameter... WHICH VALUE?
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
    # sigma = 0.05 # bandwidth

    # # @jax.jit
    # def gaussian_kernel(x1, x2):
    #     # return jnp.exp(- jnp.abs(x1 - x2) ** 2 / (2 * sigma ** 2))
    #     return jnp.exp(- jnp.abs(x1[:, jnp.newaxis] - x2) ** 2 / (2 * sigma ** 2))

    # # @jax.jit
    # def gaussian_kernel_dx(x1, x2):
    #     # return - (x1 - x2) * gaussian_kernel(x1, x2) / sigma ** 2
    #     return - (x1[:, jnp.newaxis] - x2) * gaussian_kernel(x1, x2) / sigma ** 2

    # print('Running SVGD...')

    # for s in range(steps):
    #     # if s % 10 == 0:
    #     #     print(f'  s = {s}/{steps}')
    #     # phi = lambda y: alpha * jnp.mean(gaussian_kernel(x, y) * log_mu_dx(x) + gaussian_kernel_dx(x, y))
    #     # x = x + epsilon * jnp.array([phi(x_i) for x_i in x]).reshape(-1, 1)
    #     phi = lambda y: alpha * jnp.mean(gaussian_kernel(x, y) * log_mu_dx(x) + gaussian_kernel_dx(x, y), axis=1)
    #     x = x + epsilon * phi(x)

    x = SVGD_update(x, log_mu_dx, n_iter=steps, stepsize=epsilon, alpha=alpha)

    return x