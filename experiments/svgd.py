import jax
import jax.numpy as jnp
# from scipy.spatial.distance import pdist, squareform
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
def K(x, h=0.05):
    xmx = jnp.expand_dims(x, 0) - jnp.expand_dims(x, 1)
    norm = jnp.einsum('ijk,ijk->ij', xmx, xmx)
    Kval = jnp.exp(- norm / h ** 2 / 2)
    gKval = jnp.expand_dims(Kval, -1) * (xmx / h ** 2)
    return (Kval, gKval)

#####################################################################################################

# ORIGINAL SVGD IMPLEMENTATION

@jax.jit
def SVGD_kernel(theta, h):
    '''
    Function taken from: https://github.com/dilinwang820/Stein-Variational-Gradient-Descent/blob/master/python/svgd.py
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
    Function taken from: https://github.com/dilinwang820/Stein-Variational-Gradient-Descent/blob/master/python/svgd.py
    '''
    theta = jnp.copy(x0)

    for iter in range(n_iter):
        # if debug and (iter + 1) % 1000 == 0:
        #     print('iter', str(iter + 1))
        
        lnpgrad = lnprob(theta.squeeze()).reshape(-1, 1) # lnpgrad: (n, d)
        # calculating the kernel matrix
        kxy, dxkxy = SVGD_kernel(theta, h=0.05) # kxy: (n, n), dxkxy: (n, d)
        grad_theta = alpha * (jnp.matmul(kxy, lnpgrad) + dxkxy) / x0.shape[0] # grad_theta: (n, d)
        
        # vanilla update
        theta = theta + stepsize * grad_theta

        # if iter % 10 == 0:
        #     # plot current set of particles like an animation
        #     plt.cla()
        #     # plt.plot(x_plot, gaussian(x_plot, 0, 1), color='black', label='Target distribution')
        #     plt.scatter(theta, jnp.zeros_like(theta), color='red', label='Particles')
        #     plt.legend()
        #     plt.title(f'Iteration {iter}')
        #     plt.show()
        
    return theta


def SVGD_update_vec(x0, lnprob, n_iter=1000, stepsize=1e-3, alpha=1.0, debug=False):

    def body_fn(i, theta):
        lnpgrad = lnprob(theta.squeeze()).reshape(-1, 1)
        kxy, dxkxy = SVGD_kernel(theta, h=0.05)
        grad_theta = alpha * (jnp.matmul(kxy, lnpgrad) + dxkxy) / x0.shape[0]
        theta = theta + stepsize * grad_theta
        return theta

    theta = jax.lax.fori_loop(0, n_iter, body_fn, x0)
        
    return theta


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
        L = jnp.linalg.cholesky(2 * eta * Kval + 1e-6 * jnp.eye(N))
        return (eta * (Kval @ g_logp(x.squeeze()).reshape(-1, 1) + gKval.sum(0)) + L @ jax.random.normal(key, (N, d)), 
                jax.random.split(key)[0])
    
    def body_fn(i, state):
        x, key = state
        x, key = update_svgd(x, key)
        return (x, key)
    
    x, _ = jax.lax.fori_loop(0, T, body_fn, (x0, key))
    
    return x


def test_SVGD():

    jnp.set_printoptions(linewidth=jnp.inf)
    
	# Initial particles
    x0 = jax.random.uniform(jax.random.key(0), shape=(20, ), minval=-10, maxval=10).reshape(-1, 1)
    
    dlnprob = lambda x: log_gaussian_dx(x, 0, 1)

    start = time.time()
    theta = SVGD_update(x0, dlnprob, n_iter=1000, stepsize=0.05, debug=True)
    print("Elapsed time:", time.time() - start)
    print(theta.squeeze())

    start = time.time()
    theta_vec = SVGD_update_vec(x0, dlnprob, n_iter=1000, stepsize=0.05, debug=True)
    print("Elapsed time (vectorized):", time.time() - start)
    print(theta_vec.squeeze())

    start = time.time()
    theta_new = svgd(x0, dlnprob, T=1000, eta=0.05, alpha=1.0)
    print("Elapsed time (new):", time.time() - start)
    print(theta_new.squeeze())

    start = time.time()
    theta_unbiased = svgd_unbiased(x0, dlnprob, T=1000, eta=0.05, key=jax.random.PRNGKey(0))
    print("Elapsed time (unbiased):", time.time() - start)
    print(theta_unbiased.squeeze())

    plot_on = False

    if plot_on:

        plt.scatter(x0, jnp.zeros_like(x0), color='blue', label='Initial particles')
        plt.scatter(theta, jnp.zeros_like(theta), color='red', label='Final particles')
        x_plot = jnp.linspace(-10, 10, 100)
        plt.plot(x_plot, gaussian(x_plot, 0, 1), color='black', label='Target distribution')
        plt.legend()
        plt.show()

        plt.scatter(x0, jnp.zeros_like(x0), color='blue', label='Initial particles')
        plt.scatter(theta_vec, jnp.zeros_like(theta_vec), color='red', label='Final particles')
        x_plot = jnp.linspace(-10, 10, 100)
        plt.plot(x_plot, gaussian(x_plot, 0, 1), color='black', label='Target distribution')
        plt.legend()
        plt.show()

        plt.scatter(x0, jnp.zeros_like(x0), color='blue', label='Initial particles')
        plt.scatter(theta_new, jnp.zeros_like(theta_new), color='red', label='Final particles')
        x_plot = jnp.linspace(-10, 10, 100)
        plt.plot(x_plot, gaussian(x_plot, 0, 1), color='black', label='Target distribution')
        plt.legend()
        plt.show()


# Test
test_SVGD()

#####################################################################################################

# MY SVGD IMPLEMENTATION

# @jax.jit
# def gaussian_kernel(x1, x2, sigma):
#     # return jnp.exp(- jnp.abs(x1 - x2) ** 2 / (2 * sigma ** 2))
#     return jnp.exp(- jnp.abs(x1[:, jnp.newaxis] - x2) ** 2 / (2 * sigma ** 2))

# @jax.jit
# def gaussian_kernel_dx(x1, x2, sigma):
#     # return - (x1 - x2) * gaussian_kernel(x1, x2) / sigma ** 2
#     return - (x1[:, jnp.newaxis] - x2) * gaussian_kernel(x1, x2, sigma) / sigma ** 2


# def adaptive_sampling():
#     '''
#     Adaptive sampling with SVGD.
#     '''
#     alpha = 1 # scaling parameter... WHICH VALUE?
#     epsilon = 1e-3 # step size
#     steps = 10000

#     # Initial particles
#     x0 = jax.random.uniform(jax.random.key(0), shape=(20, ), minval=-10, maxval=10)

#     # Target distribution (unnormalized)
#     mu, sigma = 0, 1
#     # gaussian = lambda x: jnp.exp(- 0.5 * (x - mu) ** 2 / sigma ** 2) # there will be a (jitted) residual here...
#     # log_gaussian = lambda x: jnp.log(gaussian(x))
#     # log_gaussian_dx = jax.vmap(jax.grad(log_gaussian), 0)

#     # Gaussian kernel
#     bdw = 0.05 # bandwidth
#     # gaussian_kernel = lambda x1, x2: jnp.exp(- jnp.abs(x1[:, jnp.newaxis] - x2) ** 2 / (2 * bdw ** 2))
#     # gaussian_kernel_dx = lambda x1, x2: - (x1[:, jnp.newaxis] - x2) * gaussian_kernel(x1, x2) / bdw ** 2

#     x = jnp.copy(x0)

#     phi = lambda y: alpha * jnp.mean(gaussian_kernel(x, y, bdw) * log_gaussian_dx(x, mu, sigma) + gaussian_kernel_dx(x, y, bdw), axis=1)

#     for s in range(steps):
#         if s % 1000 == 0:
#             print(f'  s = {s}/{steps}')
#         x = x + epsilon * phi(x)

#     print(x)

#     # Plot
#     plt.scatter(x0, jnp.zeros_like(x0), color='blue', label='Initial particles')
#     plt.scatter(x, jnp.zeros_like(x), color='red', label='Final particles')
#     x_plot = jnp.linspace(-10, 10, 100)
#     plt.plot(x_plot, gaussian(x_plot, mu, sigma), color='black', label='Target distribution')
#     plt.legend()
#     plt.show()


# Test
# adaptive_sampling()