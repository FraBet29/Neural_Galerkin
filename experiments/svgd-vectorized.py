import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


@jax.jit
def SVGD_kernel(z, h):
    z_norm_squared = jnp.sum(z ** 2, axis=1, keepdims=True)
    pairwise_dists = z_norm_squared + z_norm_squared.T - 2 * jnp.dot(z, z.T)
    Kxy = jnp.exp(- pairwise_dists / h ** 2 / 2)
    dxkxy = - jnp.matmul(Kxy, z)
    sumkxy = jnp.sum(Kxy, axis=1)
    dxkxy += jnp.multiply(z, jnp.expand_dims(sumkxy, axis=1)) # vectorized
    dxkxy /= (h ** 2)
    return (Kxy, dxkxy)


def SVGD_update(z0, log_mu_dx, steps=1000, epsilon=1e-3, alpha=1.0, diagnostic_on=False):
    '''
    Function adapted from: https://github.com/dilinwang820/Stein-Variational-Gradient-Descent/blob/master/python/svgd.py
    '''
    def body_fn(i, z):
        log_mu_dx_val = log_mu_dx(z.squeeze()).reshape(-1, 1)
        kxy, dxkxy = SVGD_kernel(z, h=0.05)
        grad_z = alpha * (jnp.matmul(kxy, log_mu_dx_val) + dxkxy) / z0.shape[0]
        z = z + epsilon * grad_z
        return z

    z = jax.lax.fori_loop(0, steps, body_fn, z0)

    return z


def SVGD_update_corrected(z0, log_mu_dx, steps=1000, epsilon=1e-3, alpha=1.0, diagnostic_on=False):
    '''
    Function adapted from: https://github.com/dilinwang820/Stein-Variational-Gradient-Descent/blob/master/python/svgd.py
    '''
    N, d = z0.shape

    def body_fn(i, state):
        z, key = state
        log_mu_dx_val = log_mu_dx(z.squeeze()).reshape(-1, 1)
        kxy, dxkxy = SVGD_kernel(z, h=0.05)
        grad_z = alpha * (jnp.matmul(kxy, log_mu_dx_val) + dxkxy) / z0.shape[0]
        # L = jnp.linalg.cholesky(2 * epsilon * kxy + 1e-3 * jnp.eye(N)) # correction needed for numerical stability
        # z = z + epsilon * grad_z + L @ jax.random.normal(key, (N, d))
        L = jnp.linalg.cholesky(kxy + 1e-6 * jnp.eye(N)) # correction needed for numerical stability
        z = z + epsilon * grad_z + jnp.sqrt(2 * epsilon / z0.shape[0]) * L @ jax.random.normal(key, (N, d))
        return (z, jax.random.split(key)[0])

    z, _ = jax.lax.fori_loop(0, steps, body_fn, (z0, jax.random.PRNGKey(0)))

    return z


# def SVGD_update(z0, log_mu_dx, steps=1000, epsilon=1e-3, alpha=1.0):
#     print('Running SVGD update...')
#     z = jnp.copy(z0)
#     z = z.reshape(-1, 1)
#     z_list = []
#     for s in range(steps):
#         log_mu_dx_val = log_mu_dx(z.squeeze()).reshape(-1, 1) # log_mu_dx: (n, d)
#         kxy, dxkxy = SVGD_kernel(z, h=0.05) # kxy: (n, n), dxkxy: (n, d)
#         grad_z = alpha * (jnp.matmul(kxy, log_mu_dx_val) + dxkxy) / z0.shape[0] # grad_x: (n, d)
#         z = z + epsilon * grad_z
#         # epsilon = epsilon * 0.9 # annealing
#         z = jnp.clip(z, 0, 1) # project back to the simplex
#         z_list.append(z)
    
#     x_plot = jnp.linspace(0, 1, 1000)
#     fig, ax = plt.subplots()
#     for idx, x in zip(range(steps), z_list):
#         plt.plot(x_plot, 1 / (x_plot + 0.1), 'r')
#         ax.scatter(x, jnp.zeros_like(z), alpha=0.1)
#         plt.title('iter: {}'.format(idx))
#         # plt.legend()
#         plt.ion()
#         plt.draw()
#         plt.show()
#         plt.pause(0.01)
#         ax.clear()
    
#     return z


n = 20
z0 = jax.random.uniform(jax.random.PRNGKey(0), (n, ))

mu = lambda y: 1 / (y + 0.1) # (n, )
log_mu = lambda y: jnp.log(mu(y)) # (n, )
log_mu_dx = jax.vmap(jax.grad(log_mu), 0) # (n, d)

z = SVGD_update(z0, log_mu_dx, steps=1000, epsilon=1e-3)