import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from wasserstein import wasserstein_1d


def gaussian(x, mu, sigma):
	return jnp.exp(- ((x - mu) / sigma) ** 2 / 2)

def gaussian_mix(x, mu, sigma, w1=0.5, w2=0.5):
	return w1 * gaussian(x, mu, sigma) + w2 * gaussian(x, - mu, sigma)

def gaussian_dx(x, mu, sigma):
	return - (x - mu) / sigma ** 2 * gaussian(x, mu, sigma)

def gaussian_mix_dx(x, mu, sigma, w1=0.5, w2=0.5):
	return w1 * gaussian_dx(x, mu, sigma) + w2 * gaussian_dx(x, - mu, sigma)

def V_dx(x, mu, sigma):
	return - gaussian_dx(x, mu, sigma) / gaussian(x, mu, sigma)

def V_dx_mix(x, mu, sigma, w1=0.5, w2=0.5):
	return - gaussian_mix_dx(x, mu, sigma, w1, w2) / gaussian_mix(x, mu, sigma, w1, w2)


# def V_linear(x, gamma=1.):
# 	return gamma * x ** 2

# def V_linear_dx(x, gamma=1.):
# 	return 2 * gamma * x

# def mu_linear(x, gamma=1.):
# 	return jnp.exp(- V_linear(x, gamma))


def high_order_runge_kutta(x, alpha=1.0, h=1e-1, T=100):

	x_list = []
	
	sigma = jnp.sqrt(2 * alpha)
	# sigma = 0.
	V_tilde = lambda x: alpha * V_dx(x, 0, 1)
	# V_tilde = lambda x: alpha * V_dx_mix(x, 3, 1, 0.3, 0.7)
	# V_tilde = lambda x: alpha * V_dx_mix(x, 2, 1, 1/3, 2/3)
	f = lambda x: - V_tilde(x)

	# sigma = 0.
	# f = lambda x: - V_linear_dx(x)

	def runge_kutta_step(x, t, h):
		csi = jax.random.normal(jax.random.key(t * 100), (x.shape[0],))
		y1 = x + jnp.sqrt(2 * h) * sigma * csi
		y2 = x - 3/8 * h * f(y1) + jnp.sqrt(2 * h) * sigma * csi / 4
		x = x - 1/3 * h * f(y1) + 4/3 * h * f(y2) + sigma * jnp.sqrt(h) * csi
		return x
	
	# def runge_kutta_step(x, t, h):
	# 	csi = jax.random.normal(jax.random.key(t * 100), (x.shape[0],))
	# 	y1 = x + sigma * jnp.sqrt(h) * csi
	# 	y2 = x - h/2 * f(y1) + sigma/2 * jnp.sqrt(h) * csi
	# 	y3 = x + 3 * h * f(y1) - 2 * h * f(y2) + sigma * jnp.sqrt(h) * csi
	# 	x = x - 3/2 * h * f(y1) + 2 * h * f(y2) + h/2 * f(y3) + sigma * jnp.sqrt(h) * csi
	# 	return x

	for t in range(T):
		if t > 0 and t % 1000 == 0:
			print(f'Iter: {t}')
		x_list.append(x)
		x = runge_kutta_step(x, t, h)
		# x = runge_kutta_step_2(x, h)
	
	return x, x_list


# def high_order_runge_kutta_vec(x, alpha=5.0, h=1e-1, T=100):

# 	# x_list = []
	
# 	sigma = jnp.sqrt(2 * alpha)
# 	# sigma = 0.
# 	# V_tilde = lambda x: alpha * V_dx(x, 0, 1)
# 	V_tilde = lambda x: alpha * V_dx_mix(x, 3, 1, 0.3, 0.7)
# 	f = lambda x: - V_tilde(x)

# 	# sigma = 0.
# 	# f = lambda x: - V_linear_dx(x)

# 	def runge_kutta_step(x, key):
# 		csi = jax.random.normal(key, (x.shape[0],))
# 		y1 = x + jnp.sqrt(2 * h) * sigma * csi
# 		y2 = x - 3/8 * h * f(y1) + jnp.sqrt(2 * h) * sigma * csi / 4
# 		x = x - 1/3 * h * f(y1) + 4/3 * h * f(y2) + sigma * jnp.sqrt(h) * csi
# 		return x, key
	
# 	def body_fn(i, state):
# 		x, key = state
# 		x, key = runge_kutta_step(x, key)
# 		return (x, key)

# 	x, _ = jax.lax.fori_loop(0, T, body_fn, (x, jax.random.key(0)))
	
# 	return x # x_list


def main():
	
	T = 10 ** 5 # time steps
	h = 0.05 # discretization step
	N = 10 # number of samples
	x0 = jax.random.normal(jax.random.key(0), (N,)) - 5
	alpha = 1.0 # noise

	x, x_list = high_order_runge_kutta(x0, alpha=alpha, h=h, T=T)
	# x = high_order_runge_kutta_vec(x0, T=T)

	# Wasserstein distance evolution
	# wass_list = []
	# for i in range(1, T):
	# 	wass_list.append(wasserstein_1d(x_list[i], x_list[i-1], p=2))
	# plt.plot(jnp.cumsum(jnp.array(wass_list)) / (jnp.arange(T - 1) + 1))
	# plt.show()

	x_plot = jnp.linspace(-10, 10, 1000)
	# y = gaussian_mix(x_plot, 3, 1, 0.3, 0.7)
	y = gaussian(x_plot, 0, 1)
	y /= y.sum()
	plt.plot(x_plot, y, label='True')
	plt.plot(x0, jnp.zeros_like(x0), 'ro', label='Initial')
	plt.plot(x, jnp.zeros_like(x), 'bo', label='Final')
	plt.legend()
	plt.show()

	# x_plot = jnp.linspace(-10, 10, 1000)
	# fig, ax = plt.subplots()
	# for idx, x in zip(range(T), x_list):
	# 	ax.plot(x_plot, gaussian(x_plot, 0, 1), label='PDF')
	# 	# ax.plot(x_plot, gaussian_mix(x_plot, 3, 1, 0.3, 0.7), label='PDF')
	# 	ax.scatter(x, jnp.zeros(N), alpha=0.1)
	# 	plt.title('iter: {}'.format(idx))
	# 	plt.legend()
	# 	plt.ion()
	# 	plt.draw()
	# 	plt.show()
	# 	plt.pause(0.01)
	# 	ax.clear()

	# Compute error for the second moment
	emp_var = jnp.zeros(N)
	for x in x_list:
		emp_var = emp_var + x ** 2
	emp_var = emp_var / T
	var = 2 * alpha # true variance
	err = jnp.abs(emp_var - var)
	print(err)


if __name__ == '__main__':
	
	# main()

	# errors with T = 10 ** 4, N = 10, alpha = 1.0
	# h_list = [0.1, 0.05, 0.01, 0.005, 0.001]
	h_list = [0.01, 0.005, 0.001]
	err_list = [# [0.9861115, 1.0164509, 0.9808866, 0.9869323, 1.0362918, 0.9489559, 0.9744103, 0.9244987, 1.0369699, 1.114398], 
			    # [0.96709096, 1.0113299, 0.9619309, 0.9813943, 1.0232048, 0.94685733, 0.9569106, 0.8621044, 1.0410488, 1.127923],
				[0.8308338, 1.1084373, 0.7893499, 0.8601471, 0.88126695, 0.8923478, 0.8615931, 0.71657264, 1.0474823, 1.0612404],
				[0.69925225, 1.1061159, 0.62426627, 0.5837655, 0.66736424, 0.7740886, 0.7242285, 0.5408536, 1.0135692, 1.0319662],
				[0.08894038, 0.3252424, 0.60443544, 1.2770636, 0.73688865, 0.19273365, 0.86919665, 1.9584563, 0.38813746, 0.5059371]]
	avg_err_list = [sum(errs) for errs in err_list]
	plt.loglog(h_list, avg_err_list, label='err')
	plt.loglog(h_list, [(h / h_list[0]) ** 2 for h in h_list], label='h^2')
	plt.legend()
	plt.show()

	# errors with T = 10 ** 5, N = 10, alpha = 1.0
	# h_list = [0.1, 0.05, 0.01, 0.005, 0.001]
	# err_list = [[0.99871755, 1.0115829, 1.0169812, 0.99996805, 1.0067, 1.006047, 1.0120919, 1.0181426, 1.0342175, 1.019784],
	# 			[0.98490894, 1.0111477, 1.0058509, 0.9743525, 0.99824893, 0.99832463, 1.0095174, 1.0072367, 1.0334196, 1.0066224]
	# 			]