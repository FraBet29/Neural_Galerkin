import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import scipy

####################################################################################################

# ENABLE PRINT ON MORE LINES
jnp.set_printoptions(threshold=jnp.inf, linewidth=jnp.inf)

# ENABLE DOUBLE PRECISION
# jax.config.update('jax_enable_x64', True)

####################################################################################################

# class for storing the results of the experiments
class Results():

	def __init__(self, p=None, v=None, v_recovered=None, v_opt=None, cond_M=None, cond_G=None, cond_G_opt=None, cond_B_GS=None,
				error=None, error_recovered=None, error_opt=None):
		self.p = p
		self.v = []
		self.v_recovered = []
		self.v_opt = []
		self.cond_M = []
		self.cond_G = []
		self.cond_G_opt = []
		self.cond_B_GS = []
		self.error = []
		self.error_recovered = []
		self.error_opt = []

	def __call__(self, show_solution=True):
		print('p =', self.p)
		if show_solution:
			print('v =', self.v)
			print('v_recovered =', self.v_recovered)
			print('v_opt =', self.v_opt)
		print('cond(M) =', self.cond_M)
		print('cond(G) =', self.cond_G)
		print('cond(G_opt) =', self.cond_G_opt)
		print('cond(B_GS) =', self.cond_B_GS)
		print('error =', self.error)
		print('error_recovered =', self.error_recovered)
		print('error_opt =', self.error_opt)
	
	def update(self, v, v_recovered, v_opt, cond_M, cond_G, cond_G_opt, cond_B_GS, error, error_recovered, error_opt):
		self.v.append(v)
		self.v_recovered.append(v_recovered)
		self.v_opt.append(v_opt)
		self.cond_M.append(cond_M)
		self.cond_G.append(cond_G)
		self.cond_G_opt.append(cond_G_opt)
		self.cond_B_GS.append(cond_B_GS)
		self.error.append(error)
		self.error_recovered.append(error_recovered)
		self.error_opt.append(error_opt)

# class for storing the orthonormal basis and the gram-schmidt change of basis
class Orth():

	def __init__(self, Q=None, B_GS=None):
		self.Q = Q
		self.B_GS = B_GS

	def __call__(self, x):
		return self.Q, self.B_GS

	def reset(self):
		self.Q = None
		self.B_GS = None

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
		z = jnp.clip(z, 0, 2 * jnp.pi) # project back to the simplex
		z_list.append(z)
	return z

####################################################################################################

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
		for _ in range(2): # reorthogonalization
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
	for i in range(p):
		for j in range(p):
			inner_prod = jnp.mean(u_dth_proj[:, i] * u_dth_proj[:, j]) * (2 * jnp.pi)
			if i != j and inner_prod > 1e-3:
				print(f'### Warning: inner product ({i}, {j}) = {inner_prod} ###')

	# recover the gram-schmidt change of basis matrix

	# B_GS_diag = jnp.sqrt(jnp.mean(A(x) ** 2, axis=0))
	B_GS_diag = vec_norm.squeeze()
	# B_GS_off = jnp.mean(u_dth_fun(x_proj)[:, :, jnp.newaxis] * u_dth_proj[:, jnp.newaxis, :], axis=0) * (2 * jnp.pi)
	B_GS_off = jnp.mean(u_dth_proj[:, :, jnp.newaxis] * u_dth_fun(x_proj)[:, jnp.newaxis, :], axis=0) * (2 * jnp.pi)
	B_GS = jnp.diag(B_GS_diag) + jnp.triu(B_GS_off, k=1).T

	return u_dth_proj, B_GS

####################################################################################################

def main(p, n, seed=0):

	# define a least squares problem where A and b are functions of x

	key = jax.random.key(seed)
	key1, key2 = jax.random.split(key)
	N = 2048

	x = 2 * jnp.pi * jax.random.uniform(key1, (n, )) # FOR NOW, SAMPLE FROM THE UNIFORM DISTRIBUTION
	x_plot = jnp.linspace(0, 2 * jnp.pi, N)

	A0 = lambda x: 1 + 0 * x
	A1 = lambda x: x
	A2 = lambda x: x ** 2
	A3 = lambda x: x ** 3
	A4 = lambda x: x ** 4
	A5 = lambda x: x ** 5
	A6 = lambda x: x ** 6
	A7 = lambda x: x ** 7
	A8 = lambda x: x ** 8
	A9 = lambda x: x ** 9

	if p == 5:
		A = lambda x: jnp.array([A0(x), A1(x), A2(x), A3(x), A4(x)]).T
	elif p == 7:
		A = lambda x: jnp.array([A0(x), A1(x), A2(x), A3(x), A4(x), A5(x), A6(x)]).T
	elif p == 10:
		A = lambda x: jnp.array([A0(x), A1(x), A2(x), A3(x), A4(x), A5(x), A6(x), A7(x), A8(x), A9(x)]).T

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

	# the normal equations are ill conditioned so it may not be a good idea to solve them directly

	M = jnp.mean(A(x)[:, :, jnp.newaxis] * A(x)[:, jnp.newaxis, :], axis=0) # (p, p)
	F = jnp.mean(A(x)[:, :] * b(x)[:, jnp.newaxis], axis=0) # (p, )

	# print('M.shape =', M.shape)
	print('cond(M) =', jnp.linalg.cond(M))

	v = jnp.linalg.solve(M, F)

	# try to solve the least squares problem directly

	v_check = jnp.linalg.lstsq(A(x), b(x))[0]

	# print('error (normal equations check pt. 1) =', jnp.linalg.norm(jnp.dot(A(x_plot), v) - b(x_plot)))
	print('error (normal equations check pt. 1) =', jnp.mean((jnp.dot(A(x_plot), v) - b(x_plot)) ** 2))

	# print('error (normal equations check pt. 2) =', jnp.linalg.norm(jnp.dot(A(x_plot), v_check) - b(x_plot)))
	print('error (normal equations check pt. 2) =', jnp.mean((jnp.dot(A(x_plot), v_check) - b(x_plot)) ** 2))


	# we want to orthogonalize the columns of A to form an orthonormal basis for the column space of A
	# to do so we use gram-schmidt with the L^2 inner product
	# for this step, we use sample points x_proj that are different from x

	# compute an orthonormal basis to define weighted estimators

	def w_fn(x, store): # returns the weights, the orthonormal basis, and the Gram-Schmidt basis
		
		# compute orthonormal basis evaluated on x_proj
		x_proj = 2 * jnp.pi * jax.random.uniform(key2, (N, ))
		x_proj = jnp.sort(x_proj) # needed for interpolation

		if store.Q is None:
			print('Orthogonalizing...')
			Q_proj, B_GS = orthogonalize(x_proj, A)
			store.Q = Q_proj
			store.B_GS = B_GS
		else:
			Q_proj, B_GS = store.Q, store.B_GS

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


	store = Orth() # store the orthonormal basis and the gram-schmidt change of basis to avoid recomputing them

	w, Q, B_GS = w_fn(x, store)
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

	# print('error (normal equations check pt. 3) =', jnp.linalg.norm(jnp.dot(A(x_plot), v_recovered) - b(x_plot)))
	print('error (normal equations check pt. 3) =', jnp.mean((jnp.dot(A(x_plot), v_recovered) - b(x_plot)) ** 2))

	print('v =', v)
	print('v_recovered =', v_recovered)

	# we can also draw samples from a weighted distribution
	# if we use optimal weights we can ensure that the conditioning number of the normal equations is small

	# problem: need to sample from the weighted distribution... need SVGD

	mu = lambda y: 1 / w_fn(y, store)[0] # (n, )
	mu_dx = jax.vmap(jax.grad(mu), 0) # (n, d)
	log_mu = lambda y: jnp.log(mu(y)) # (n, )
	log_mu_dx = jax.vmap(jax.grad(log_mu), 0) # (n, d)

	x_opt = SVGD_update(x, mu, log_mu, log_mu_dx, steps=500, epsilon=0.5).squeeze()

	# check sampling
	# plt.plot(x_plot, mu(x_plot))
	# plt.plot(x_opt, jnp.zeros_like(x_opt), 'ro')
	# plt.show()

	w_opt, Q_opt, B_GS_opt = w_fn(x_opt, store)

	G_opt = jnp.mean(w_opt[: , jnp.newaxis, jnp.newaxis] * (Q_opt[:, :, jnp.newaxis] * Q_opt[:, jnp.newaxis, :]), axis=0) # (p, p)
	d_opt = jnp.mean(w_opt[:, jnp.newaxis] * (Q_opt[:, :] * b(x_opt)[:, jnp.newaxis]), axis=0) # (p, )

	print('cond(G_opt) =', jnp.linalg.cond(G_opt))

	v_weighted_opt = jnp.linalg.solve(G_opt, d_opt)

	print('error (opt) =', jnp.linalg.norm(jnp.dot(G_opt, v_weighted_opt) - d_opt))

	v_opt = jnp.linalg.solve(B_GS_opt.T, v_weighted_opt)
	# print('error (normal equations check pt. 4) =', jnp.linalg.norm(jnp.dot(A(x_plot), v_opt) - b(x_plot)))
	print('error (normal equations check pt. 4) =', jnp.mean((jnp.dot(A(x_plot), v_opt) - b(x_plot)) ** 2))

	print('v_opt =', v_opt)

	# plt.figure()
	# plt.plot(v, 'o-', label='v')
	# plt.plot(v_recovered, 'o-', label='v_recovered')
	# plt.plot(v_opt, 'o-', label='v_opt')
	# plt.legend()
	# plt.show()

	return v, v_recovered, v_opt, jnp.linalg.cond(M), jnp.linalg.cond(G), jnp.linalg.cond(G_opt), jnp.linalg.cond(B_GS), \
			jnp.mean((jnp.dot(A(x_plot), v) - b(x_plot)) ** 2), jnp.mean((jnp.dot(A(x_plot), v_recovered) - b(x_plot)) ** 2), \
			jnp.mean((jnp.dot(A(x_plot), v_opt) - b(x_plot)) ** 2)

	# if we suppose instead that the columns of A are linearly dependent we can still orthogonalize by discarding the dependent columns
	# we can use the same weighted estimators as before if we are able to estimate the rank of the column space of A
	# again, the optimal weights ensure that the conditioning number of the normal equations is small

####################################################################################################

if __name__ == '__main__':

	p = 10 # number of basis functions
	n = 1000 # number of sample points

	results = Results(p)

	for i in range(10):
		print(f'Running experiment {i + 1}...')
		v, v_recovered, v_opt, cond_M, cond_G, cond_G_opt, cond_B_GS, error, error_recovered, error_opt = main(p, n, seed=int(i * 1e3))
		results.update(v, v_recovered, v_opt, cond_M, cond_G, cond_G_opt, cond_B_GS, error, error_recovered, error_opt)

	results(show_solution=False)

	# RESULTS with p = 5, n = 100

	cond_M = [
		63371672.0, 55786696.0, 89176936.0, 71845032.0, 139972320.0, 
		45917520.0, 50165412.0, 85356136.0, 55873804.0, 53289712.0
	]
	cond_G = [
		3.2138507, 5.080805, 3.3012092, 3.3720467, 4.182983, 
		4.2012706, 4.4523745, 4.6362076, 5.5976954, 4.5063767
	]
	cond_G_opt = [
		1.4367008, 1.4515696, 1.6161714, 1.5083665, 1.5124675, 
		1.5214396, 1.552127, 1.4953238, 1.4728917, 1.5932888
	]
	cond_B_GS = [
		7351.361, 7331.133, 7404.0127, 7538.6475, 7440.377, 
		7419.8506, 6789.327, 7362.826, 7114.2456, 7095.6807
	]
	error = [
		0.00449639, 0.00455858, 0.00490514, 0.00450945, 0.00571216, 
		0.00473317, 0.0059745, 0.00490718, 0.00470452, 0.00487894
	]
	error_recovered = [
		0.00554369, 0.00655066, 0.00632325, 0.00607112, 0.00843527, 
		0.00686536, 0.00730204, 0.00666687, 0.0056615, 0.00643466
	]
	error_opt = [
		0.00458381, 0.00459823, 0.00469071, 0.00473881, 0.00456011, 
		0.00466656, 0.00464439, 0.00459668, 0.00465596, 0.00466028
	]

	print('p = 5, n = 100')
	print('Mean error:', jnp.mean(jnp.array(error)))
	print('Mean error (recovered):', jnp.mean(jnp.array(error_recovered)))
	print('Mean error (opt):', jnp.mean(jnp.array(error_opt)))
	print('P[k(G_opt) < 3] = ', jnp.mean(jnp.array(cond_G_opt) < 3))

	# RESULTS with p = 7, n = 100

	cond_M = [
		6.669712e+11, 2.4574177e+10, 8.587665e+10, 1.1779716e+11, 1.1658691e+11,
		6.3524844e+11, 3.0536358e+10, 2.4436687e+10, 6.341553e+11, 7.346612e+10
	]
	cond_G = [
		5.7243233, 7.180855, 4.8099284, 4.469931, 8.524943,
		8.45836, 11.530276, 7.9887886, 7.782873, 7.601114
	]
	cond_G_opt = [
		1.7649835, 1.6706486, 2.0729835, 1.9726442, 1.8746979,
		1.7231567, 1.9734071, 1.9434997, 1.7880076, 2.0318255
	]
	cond_B_GS = [
		1007351.0, 988681.44, 1001940.7, 1053154.9, 1051934.0,
		1033455.5, 946023.75, 1025526.75, 1027984.75, 989286.4
	]
	error = [
		0.00013041, 0.00010497, 0.00012813, 0.00015417, 0.00037552,
		0.0040254, 0.00027147, 0.00025072, 0.00017062, 7.689763e-05
	]
	error_recovered = [
		3.088681e-05, 3.1332118e-05, 2.6151516e-05, 2.3407065e-05, 4.3849304e-05,
		3.3744516e-05, 7.152688e-05, 3.6557016e-05, 3.2858345e-05, 3.5544956e-05
	]
	error_opt = [
		2.1247455e-05, 2.2371038e-05, 2.5327248e-05, 2.1339176e-05, 2.2655988e-05,
		2.0565347e-05, 2.2540476e-05, 2.3077806e-05, 2.1551947e-05, 2.1677066e-05
	]

	print('p = 7, n = 100')
	print('Mean error:', jnp.mean(jnp.array(error)))
	print('Mean error (recovered):', jnp.mean(jnp.array(error_recovered)))
	print('Mean error (opt):', jnp.mean(jnp.array(error_opt)))
	print('P[k(G_opt) < 3] = ', jnp.mean(jnp.array(cond_G_opt) < 3))

	# RESULTS with p = 10, n = 100 (WARNING: loss of orthogonality)

	cond_M = [
		2.2307808e+14, 8.783196e+14, 7.656043e+14, 1.1881147e+15, 2.0021062e+14,
		1.7833907e+15, 3.1811768e+15, 4.0269674e+14, 2.6702343e+15, 2.5230626e+14
	]
	cond_G = [
		9.6794, 13.09234, 9.063999, 5.113416, 23.504595,
		14.793338, 79.78598, 16.647102, 19.631256, 16.645319
	]
	cond_G_opt = [
		7.2399583, 4.001335, 8.350032, 7.335317, 3.7498255,
		7.3501053, 5.346324, 8.510305, 4.652231, 3.800225
	]
	cond_B_GS = [
		1.1665417e+11, 1.2562284e+10, 1.2309612e+10, 8.788481e+09, 6.649881e+09,
		1.4481252e+11, 1.2785005e+10, 1.5470398e+10, 1.03138525e+11, 2.3193287e+10
	]
	error = [
		0.00574128, 1.4640687e-05, 2.1638212e-05, 8.3913605e-05,
		0.00031122, 1.7999111e-05, 0.00019089, 0.00030701, 0.0001556, 1.8982334e-06
	]
	error_recovered = [
		0.00497671, 0.0004962, 0.00059952, 3.7374123e-06, 3.6120618e-06,
		0.00265899, 8.797847e-06, 1.30221015e-05, 0.00037394, 5.6463883e-05
	]
	error_opt = [
		0.00058081, 2.7725491e-05, 0.00049804, 1.0611132e-06, 8.5693483e-07,
		0.0008842, 4.6712876e-07, 3.5565927e-07, 3.7664537e-05, 1.8451014e-05
	]

	print('p = 10, n = 100')
	print('Mean error:', jnp.mean(jnp.array(error)))
	print('Mean error (recovered):', jnp.mean(jnp.array(error_recovered)))
	print('Mean error (opt):', jnp.mean(jnp.array(error_opt)))
	print('P[k(G_opt) < 3] = ', jnp.mean(jnp.array(cond_G_opt) < 3))

	# RESULTS with p = 10, n = 1000 (WARNING: loss of orthogonality)

	cond_M = [
		3.6774503e+14, 5.3528248e+14, 6.5984146e+14, 2.622326e+14, 7.5790395e+14,
		1.959209e+16, 1.14850655e+14, 4.637999e+14, 5.0642946e+14, 3.012213e+14
	]
	cond_G = [
		6.5446134, 7.459116, 7.612048, 5.1403527, 6.0791893,
		5.392812, 5.8135405, 6.3215337, 5.630885, 6.7155986
	]
	cond_G_opt = [
		2.6012728, 4.46289, 2.5980704, 2.725597, 2.6342564,
		2.3211205, 2.444346, 2.9474769, 4.0239267, 2.7781303
	]
	cond_B_GS = [
		1.1665417e+11, 1.2562284e+10, 1.2309612e+10, 8.788481e+09, 6.649881e+09,
		1.4481252e+11, 1.2785005e+10, 1.5470398e+10, 1.03138525e+11, 2.3193287e+10
	]
	error = [
		4.011755e-06, 5.9144677e-06, 7.71028e-06, 8.999998e-05, 4.9279315e-05,
		0.00039285, 2.8307213e-05, 8.380123e-07, 0.00019855, 0.00090952
	]
	error_recovered = [
		0.00477516, 0.0004791, 0.00058589, 3.7082725e-06, 3.275484e-06,
		0.00264639, 8.653392e-06, 1.2490597e-05, 0.00036297, 5.647944e-05
	]
	error_opt = [
		0.00088537, 6.572209e-05, 0.00112367, 1.1239239e-06, 8.021451e-07,
		0.00086771, 6.7169043e-07, 8.366722e-07, 4.4195483e-05, 1.6056805e-05
	]

	print('p = 10, n = 1000')
	print('Mean error:', jnp.mean(jnp.array(error)))
	print('Mean error (recovered):', jnp.mean(jnp.array(error_recovered)))
	print('Mean error (opt):', jnp.mean(jnp.array(error_opt)))
	print('P[k(G_opt) < 3] = ', jnp.mean(jnp.array(cond_G_opt) < 3))
