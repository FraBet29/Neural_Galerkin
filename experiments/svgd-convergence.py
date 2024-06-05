import jax
import jax.numpy as jnp
import scipy.stats
import matplotlib.pyplot as plt
import time
from wasserstein import wasserstein_1d


@jax.jit
def gaussian(x, mu, sigma):
    return jnp.exp(- (x - mu) ** 2 / sigma ** 2)

@jax.jit
def log_gaussian(x, mu, sigma):
    return jnp.log(gaussian(x, mu, sigma))

@jax.jit
def log_gaussian_dx(x, mu, sigma):
    return jax.vmap(jax.grad(log_gaussian), (0, None, None))(x, mu, sigma)


@jax.jit
def SVGD_kernel(z, h=0.05):
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
        kxy, dxkxy = SVGD_kernel(z, 0.05) # kxy: (n, n), dxkxy: (n, d)
        grad_z = alpha * (jnp.matmul(kxy, log_mu_dx_val) + dxkxy) / z0.shape[0] # grad_x: (n, d)
        # Vanilla update
        z = z + epsilon * grad_z

    return z


#####################################################################################################

# def K(x, h=0.05):
#     xmx = jnp.expand_dims(x, 0) - jnp.expand_dims(x, 1)
#     norm = jnp.einsum('ijk,ijk->ij', xmx, xmx)
#     return jnp.exp(-(norm) / h)


# def g_K(x, h=0.05):
#     # we avoid calling autograd since the function is non-scalar, for better efficiency
#     xmx = jnp.expand_dims(x, 0) - jnp.expand_dims(x, 1)
#     return jnp.expand_dims(K(x), -1) * (2.*xmx/h)


# def logp(x):
#     # Standard Gaussian target
#     return -jnp.sum(x**2)


# def svgd(x0, logp, T=100, eta=0.01, alpha=1.0):
    
#     x = x0
#     g_logp = jax.grad(logp)

#     update_svgd = jax.jit(lambda x: alpha * (K(x) @ g_logp(x) + g_K(x).sum(0))) 

#     for i in range(T):
#         x = x + eta * update_svgd(x)   
#     return x


# def sgldr(x0, logp, T=100, eta=0.01, alpha=1.0, key=jax.random.PRNGKey(0)):
    
#     N, d = x0.shape
#     x = x0
#     g_logp = jax.grad(logp)
#     xs = []

#     update_sgldr = jax.jit(lambda x, key: (eta * alpha * (K(x) @ g_logp(x) + g_K(x).sum(0)) + \
#                                           jnp.linalg.cholesky(2 * eta * K(x)) @ jax.random.normal(key, (N, d)), jax.random.split(key)[0]))

#     for i in range(T):
#         x_upd, key = update_sgldr(x, key)
#         x = x + x_upd
#         xs.append(x)
#     return x, xs

#####################################################################################################

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstest.html
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.goodness_of_fit.html

def test_SVGD():

    N = 1000 # number of particles
    
	# Initial particles
    x0 = jax.random.uniform(jax.random.key(0), shape=(N, ), minval=-10, maxval=10).reshape(-1, 1)

    # Goodness-of-fit test for initial particles
    x_plot = jnp.linspace(-10, 10, 1000)
    init_gof = scipy.stats.kstest(x0.squeeze(), gaussian(x_plot, 0, 1).squeeze()) # Kolmogorov-Smirnov test
    print(f'Initial goodness-of-fit: {init_gof}')

    # Wasserstein distance for initial particles
    wass_init = wasserstein_1d(x0.squeeze(), gaussian(x_plot, 0, 1).squeeze(), p=1)
    print(f'Initial Wasserstein distance: {wass_init}')

    # Plot initial histogram vs target distribution
    # hist = jnp.histogram(x0)
    # hist_dist = scipy.stats.rv_histogram(hist, density=True)
    # # plt.hist(x0, density=True)
    # plt.plot(x_plot, hist_dist.pdf(x_plot), color='red')
    # plt.plot(x_plot, gaussian(x_plot, 0, 1), color='black')
    # plt.show()
    
    log_mu_dx = lambda x: log_gaussian_dx(x, 0, 1)

    x = SVGD_update(x0, log_mu_dx, steps=100, epsilon=0.01, alpha=1.0)
    # x = svgd(x0, logp, T=100, eta=0.05, alpha=1.0)
    # x, xs = sgldr(x0, logp, T=100, eta=0.05, alpha=1.0)

    # Plot
    plt.scatter(x0, jnp.zeros_like(x0), color='blue', label='Initial particles')
    plt.scatter(x, jnp.zeros_like(x), color='red', label='Final particles')
    plt.plot(x_plot, gaussian(x_plot, 0, 1), color='black', label='Target distribution')
    plt.legend()
    plt.show()

    # Goodness-of-fit test for final particles
    final_gof = scipy.stats.kstest(x.squeeze(), gaussian(x_plot, 0, 1).squeeze())
    print(f'Final goodness-of-fit: {final_gof}')

    # Plot final histogram vs target distribution
    # hist = jnp.histogram(x)
    # hist_dist = scipy.stats.rv_histogram(hist, density=True)
    # # plt.hist(x, density=True)
    # x_plot = jnp.linspace(-10, 10, 1000)
    # plt.plot(x_plot, hist_dist.pdf(x_plot), color='red')
    # plt.plot(x_plot, gaussian(x_plot, 0, 1), color='black')
    # plt.show()

    # Wasserstein distance for final particles
    wass = wasserstein_1d(x0.squeeze(), x.squeeze(), p=1)
    print(f'Final Wasserstein distance: {wass}')


# Test
test_SVGD()