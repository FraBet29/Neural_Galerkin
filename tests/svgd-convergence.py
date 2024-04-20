import jax
import jax.numpy as jnp
import scipy.stats
import matplotlib.pyplot as plt
import time


@jax.jit
def gaussian(x, mu, sigma):
    return jnp.exp(- 0.5 * (x - mu) ** 2 / sigma ** 2)

@jax.jit
def log_gaussian(x, mu, sigma):
    return jnp.log(gaussian(x, mu, sigma))

@jax.jit
def log_gaussian_dx(x, mu, sigma):
    return jax.vmap(jax.grad(log_gaussian), (0, None, None))(x, mu, sigma)


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

        # if s % 10 == 0:
        # # plot current set of particles like an animation
        #     plt.cla()
        #     # plt.plot(x_plot, gaussian(x_plot, 0, 1), color='black', label='Target distribution')
        #     plt.scatter(z, jnp.zeros_like(z), color='red', label='Particles')
        #     plt.legend()
        #     plt.title(f'Iteration {s}')
        #     plt.show()
    
    # print(f'SVGD iterations: {s + 1}')

    return z


# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstest.html
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.goodness_of_fit.html

def test_SVGD():

    N = 10000 # number of particles
    
	# Initial particles
    x0 = jax.random.uniform(jax.random.key(0), shape=(N, ), minval=-10, maxval=10).reshape(-1, 1)

    # Goodness-of-fit test for initial particles
    x_plot = jnp.linspace(-10, 10, 1000)
    init_gof = scipy.stats.kstest(x0.squeeze(), gaussian(x_plot, 0, 1).squeeze()) # Kolmogorov-Smirnov test
    print(f'Initial goodness-of-fit: {init_gof}')

    # Plot initial histogram vs target distribution
    # hist = jnp.histogram(x0)
    # hist_dist = scipy.stats.rv_histogram(hist, density=True)
    # # plt.hist(x0, density=True)
    # plt.plot(x_plot, hist_dist.pdf(x_plot), color='red')
    # plt.plot(x_plot, gaussian(x_plot, 0, 1), color='black')
    # plt.show()
    
    log_mu_dx = lambda x: log_gaussian_dx(x, 0, 1)

    x = SVGD_update(x0, log_mu_dx, steps=500, epsilon=0.05, alpha=30.0)

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


# Test
test_SVGD()