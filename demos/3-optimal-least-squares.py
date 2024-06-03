import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import scipy


# ENABLE PRINT ON MORE LINES
jnp.set_printoptions(threshold=jnp.inf, linewidth=jnp.inf)

# ENABLE DOUBLE PRECISION
# jax.config.update('jax_enable_x64', True)


# define a least squares problem min ||Ax - b||^2 where A and b are functions of x

n = 50
p = 5
key = jax.random.key(0)
N = 2048

# A = 1./(jnp.arange(1, p+1) + jnp.arange(1, n+1)[:, jnp.newaxis])
# b = jax.random.normal(key, (n, ))

x = 2 * jnp.pi * jax.random.uniform(key, (n, )) # FOR NOW, SAMPLE FROM THE UNIFORM DISTRIBUTION
x_plot = jnp.linspace(0, 2 * jnp.pi, N)

A0 = lambda x: 1 + 0 * x
A1 = lambda x: x
A2 = lambda x: x ** 2
A3 = lambda x: x ** 3
A4 = lambda x: x ** 4
A = lambda x: jnp.array([A0(x), A1(x), A2(x), A3(x), A4(x)]).T

b = lambda x: jnp.sin(x)

# plot the original vectors

# plt.figure()
# plt.plot(x_plot, A0(x_plot), label='A_0(x)')
# plt.plot(x_plot, A1(x_plot), label='A_1(x)')
# plt.plot(x_plot, A2(x_plot), label='A_2(x)')
# plt.plot(x_plot, A3(x_plot), label='A_3(x)')
# plt.plot(x_plot, A4(x_plot), label='A_4(x)')
# plt.legend()
# plt.show()

# suppose first that the columns of A are linearly independent

# print('A.shape =', A(x).shape)
# print('b.shape =', b(x).shape)
print('rank(A) =', jnp.linalg.matrix_rank(A(x)))

# try to solve the least squares problem directly

# v = jnp.linalg.lstsq(A(x), b(x))[0]
# print('error (original) =', jnp.linalg.norm(jnp.dot(A(x_plot), v) - b(x_plot)))

# the associated normal equations are ill conditioned so it is not a good idea to solve them directly

M = jnp.mean(A(x)[:, :, jnp.newaxis] * A(x)[:, jnp.newaxis, :], axis=0) # (p, p)
F = jnp.mean(A(x)[:, :] * b(x)[:, jnp.newaxis], axis=0) # (p, )

# print('M.shape =', M.shape)
print('cond(M) =', jnp.linalg.cond(M))

v = jnp.linalg.solve(M, F)

print('error (normal equations) =', jnp.linalg.norm(jnp.dot(M, v) - F))

# sanity check for the normal equations

v_check = jnp.linalg.lstsq(A(x), b(x))[0]

print('error (normal equations check pt. 1) =', jnp.linalg.norm(jnp.dot(A(x_plot), v) - b(x_plot)))
print('error (normal equations check pt. 2) =', jnp.linalg.norm(jnp.dot(A(x_plot), v_check) - b(x_plot)))


# we want to orthogonalize the columns of A to form an orthonormal basis for the column space of A
# to do so we use gram-schmidt with the L^2 inner product
# for this step, we use sample points x_proj that are different from x


# @jax.jit
def orthogonalize(x_proj, u_dth_fun):
	'''
	Implementation of the modified Gram-Schmidt orthonormalization algorithm.
	Adapted from: https://github.com/tensorflow/probability/blob/v0.23.0/tensorflow_probability/python/math/gram_schmidt.py#L28
	'''
	# u_dth = u_dth_fun(x)
	u_dth_proj = u_dth_fun(x_proj) # (n_proj, p)
	p = u_dth_proj.shape[-1]

	def body_fn(i, vecs): # the initial vectors in vecs are progressively replaced by the orthonormalized ones
		vec_norm = jnp.sqrt(jnp.mean(vecs[:, i] ** 2) * (2 * jnp.pi))
		u = jnp.divide(vecs[:, i], vec_norm) # (n_proj, )
		weights = jnp.mean(u[:, jnp.newaxis] * vecs, axis=0) * (2 * jnp.pi) # (p, )
		# weights = jnp.mean(u[:, jnp.newaxis] * vecs, axis=0) / jnp.mean(vecs * vecs, axis=0) # (p, ) # NO?
		masked_weights = jnp.where(jnp.arange(p) > i, weights, 0.)[jnp.newaxis, :] # (1, p) # consider only the first i vectors
		vecs = vecs - jnp.multiply(u[:, jnp.newaxis], masked_weights) # (n_proj, p)
		vecs = jnp.where(jnp.isnan(vecs), 0.0, vecs)
		vecs = jnp.reshape(vecs, u_dth_proj.shape)
		return vecs

	u_dth_proj = jax.lax.fori_loop(0, p, body_fn, u_dth_proj)
	vec_norm = jnp.sqrt(jnp.mean(u_dth_proj * u_dth_proj, axis=0) * (2 * jnp.pi))
	u_dth_proj = jnp.divide(u_dth_proj, vec_norm)
	u_dth_proj = jnp.where(jnp.isnan(u_dth_proj), 0.0, u_dth_proj)

	# orthonormality check
	# for i in range(p):
	# 	for j in range(p):
	# 		print(f'inner product ({i}, {j}) =', jnp.mean(u_dth_proj[:, i] * u_dth_proj[:, j]) * (2 * jnp.pi))

	# recover the gram-schmidt change of basis matrix

	# B_GS_diag = jnp.sqrt(jnp.mean(A(x) ** 2, axis=0))
	B_GS_diag = vec_norm.squeeze()
	# B_GS_off = jnp.mean(u_dth_fun(x_proj)[:, :, jnp.newaxis] * u_dth_proj[:, jnp.newaxis, :], axis=0) * (2 * jnp.pi)
	B_GS_off = jnp.mean(u_dth_proj[:, :, jnp.newaxis] * u_dth_fun(x_proj)[:, jnp.newaxis, :], axis=0) * (2 * jnp.pi)
	B_GS = jnp.diag(B_GS_diag) + jnp.triu(B_GS_off, k=1).T

	return u_dth_proj, B_GS


# compute an orthonormal basis to define weighted estimators

def w_fn(x): # returns the weights, the orthonormal basis, and the Gram-Schmidt basis
	# compute orthonormal basis evaluated on x_proj
	x_proj = 2 * jnp.pi * jax.random.uniform(key, (N, ))
	x_proj = jnp.sort(x_proj) # needed for interpolation
	Q_proj, B_GS = orthogonalize(x_proj, A)

	if x.ndim == 0:
		n = 1
	else:
		n = x.shape[0]

	# evaluate the orthonormal basis on x via interpolation
	Q = jnp.zeros((n, p))
	for i in range(p):
		Q = Q.at[:, i].set(jnp.interp(x, x_proj, Q_proj[:, i]))

	# plot the orthonormal basis
	# plt.figure()
	# plt.plot(x_proj, Q_proj[:, 0], label='Q_0')
	# plt.plot(x_proj, Q_proj[:, 1], label='Q_1')
	# plt.plot(x_proj, Q_proj[:, 2], label='Q_2')
	# plt.plot(x_proj, Q_proj[:, 3], label='Q_3')
	# plt.plot(x_proj, Q_proj[:, 4], label='Q_4')
	# plt.scatter(x, Q[:, 0], label='Q_0(x)')
	# plt.scatter(x, Q[:, 1], label='Q_1(x)')
	# plt.scatter(x, Q[:, 2], label='Q_2(x)')
	# plt.scatter(x, Q[:, 3], label='Q_3(x)')
	# plt.scatter(x, Q[:, 4], label='Q_4(x)')
	# plt.legend()
	# plt.show()

	# compare with the analytical solution (Legendre polynomials in [0, 2pi])
	# plt.figure()
	# plt.plot(x_proj, P0(x_proj), label='P_0')
	# plt.plot(x_proj, P1(x_proj), label='P_1')
	# plt.plot(x_proj, P2(x_proj), label='P_2')
	# plt.plot(x_proj, P3(x_proj), label='P_3')
	# plt.plot(x_proj, P4(x_proj), label='P_4')
	# plt.legend()
	# plt.show()
	
	return (p / jnp.sum(Q ** 2, axis=1)).squeeze(), Q, B_GS # (n, )


w, Q, B_GS = w_fn(x)
# print('w.shape =', w.shape)
# print('Q.shape =', Q.shape)
# print('B_GS.shape =', B_GS.shape)

G = jnp.mean(w[: , jnp.newaxis, jnp.newaxis] * (Q[:, :, jnp.newaxis] * Q[:, jnp.newaxis, :]), axis=0) # (p, p)
d = jnp.mean(w[:, jnp.newaxis] * (Q[:, :] * b(x)[:, jnp.newaxis]), axis=0) # (p, )

# print('G.shape =', G.shape)
# print('d.shape =', d.shape)

print('cond(G) =', jnp.linalg.cond(G))

v_weighted = jnp.linalg.solve(G, d)

print('error (weighted) =', jnp.linalg.norm(jnp.dot(G, v_weighted) - d))

# recover the original solution

print('cond(B_GS) =', jnp.linalg.cond(B_GS))

v_recovered = jnp.linalg.solve(B_GS.T, v_weighted)
print('error (normal equations check pt. 3) =', jnp.linalg.norm(jnp.dot(A(x_plot), v_recovered) - b(x_plot)))

print('v =', v)
print('v_recovered =', v_recovered)


####################################################################################################

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


def SVGD_update(z0, mu, log_mu, log_mu_dx, steps=1000, epsilon=1e-3, alpha=1.0):
    print('Running SVGD update...')
    z = jnp.copy(z0)
    z = z.reshape(-1, 1)
    z_list = []
    for s in range(steps):
        log_mu_dx_val = log_mu_dx(z.squeeze()).reshape(-1, 1) # log_mu_dx: (n, d)
        kxy, dxkxy = SVGD_kernel(z, h=0.05) # kxy: (n, n), dxkxy: (n, d)
        grad_z = alpha * (jnp.matmul(kxy, log_mu_dx_val) + dxkxy) / z0.shape[0] # grad_x: (n, d)
        z = z + epsilon * grad_z
        # epsilon = epsilon * 0.9 # annealing
        # z = jnp.clip(z, 0, 2 * jnp.pi) # project back to the simplex
        z_list.append(z)
    
    # x_plot = jnp.linspace(0, 2 * jnp.pi, N)
    # fig, ax = plt.subplots()
    # for idx, z in zip(range(steps), z_list):
    #     plt.plot(x_plot, mu(x_plot), label='mu(x)')
    #     # plt.plot(x_plot, log_mu(x_plot), label='log(mu(x))')
    #     ax.scatter(z, jnp.zeros_like(z), alpha=0.1)
    #     plt.title('iter: {}'.format(idx))
    #     # plt.legend()
    #     plt.ion()
    #     plt.draw()
    #     plt.show()
    #     plt.pause(0.01)
    #     ax.clear()
    
    return z


def high_order_runge_kutta(x, mu, mu_dx, alpha=0.1, h=1e-1, T=100):
	print('Running high-order Runge-Kutta...')

	x_list = []
	
	sigma = jnp.sqrt(2 * alpha)
	V_dx = lambda x: - alpha * mu_dx(x) / mu(x)
	f = lambda x: V_dx(x)

	def runge_kutta_step(x, t, h):
		csi = jax.random.normal(jax.random.PRNGKey(t * 100), (x.shape[0],))
		y1 = x + jnp.sqrt(2 * h) * sigma * csi
		y2 = x - 3/8 * h * f(y1) + jnp.sqrt(2 * h) * sigma * csi / 4
		x = x - 1/3 * h * f(y1) + 4/3 * h * f(y2) + sigma * jnp.sqrt(h) * csi
		return x

	for t in range(T):
		x = runge_kutta_step(x, t, h)
		# x = jnp.clip(x, 0, 2 * jnp.pi) # project back to the simplex
		x_list.append(x)

	# x_plot = jnp.linspace(0, 2 * jnp.pi, N)
	# fig, ax = plt.subplots()
	# for idx, x in zip(range(T), x_list):
	# 	plt.plot(x_plot, mu(x_plot), label='mu(x)')
	# 	# plt.plot(x_plot, log_mu(x_plot), label='log(mu(x))')
	# 	ax.scatter(x, jnp.zeros_like(x), alpha=0.1)
	# 	plt.title('iter: {}'.format(idx))
	# 	# plt.legend()
	# 	plt.ion()
	# 	plt.draw()
	# 	plt.show()
	# 	plt.pause(0.01)
	# 	ax.clear()
	
	return x

####################################################################################################


# # we can also draw samples from a weighted distribution
# # if we use optimal weights we can ensure that the conditioning number of the normal equations is small

# # problem: need to sample from the weighted distribution... need SVGD

mu = lambda y: 1 / w_fn(y)[0] # (n, )
mu_dx = jax.vmap(jax.grad(mu), 0) # (n, d)
log_mu = lambda y: jnp.log(mu(y)) # (n, )
log_mu_dx = jax.vmap(jax.grad(log_mu), 0) # (n, d)

x_opt = SVGD_update(x, mu, log_mu, log_mu_dx, steps=100, epsilon=0.5).squeeze()
# x_opt = high_order_runge_kutta(x, mu, mu_dx, alpha=0.01, h=0.1, T=300)

# check sampling
plt.plot(x_plot, mu(x_plot))
plt.plot(x_opt, jnp.zeros_like(x_opt), 'ro')
plt.show()

w_opt, Q_opt, B_GS_opt = w_fn(x_opt)

G_opt = jnp.mean(w_opt[: , jnp.newaxis, jnp.newaxis] * (Q_opt[:, :, jnp.newaxis] * Q_opt[:, jnp.newaxis, :]), axis=0) # (p, p)
d_opt = jnp.mean(w_opt[:, jnp.newaxis] * (Q_opt[:, :] * b(x_opt)[:, jnp.newaxis]), axis=0) # (p, )

print('cond(G_opt) =', jnp.linalg.cond(G_opt))

v_weighted_opt = jnp.linalg.solve(G_opt, d_opt)

print('error (opt) =', jnp.linalg.norm(jnp.dot(G_opt, v_weighted_opt) - d_opt))
print('cond(B_GS_opt) =', jnp.linalg.cond(B_GS_opt))

v_opt = jnp.linalg.solve(B_GS_opt.T, v_weighted_opt)
print('error (normal equations check pt. 4) =', jnp.linalg.norm(jnp.dot(A(x_plot), v_opt) - b(x_plot)))


# if we suppose instead that the columns of A are linearly dependent we can still orthogonalize by discarding the dependent columns
# we can use the same weighted estimators as before if we are able to estimate the rank of the column space of A
# again, the optimal weights ensure that the conditioning number of the normal equations is small
