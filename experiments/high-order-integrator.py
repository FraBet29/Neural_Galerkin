import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


def gaussian(x, mu, sigma):
	return jnp.exp(-0.5 * ((x - mu) / sigma)**2) / jnp.sqrt(2 * jnp.pi * sigma**2)

def gaussian_mix(x, mu, sigma, w1=0.5, w2=0.5):
	return w1 * gaussian(x, mu, sigma) + w2 * gaussian(x, - mu, sigma)

def gaussian_dx(x, mu, sigma):
	return - (x - mu) / sigma**2 * gaussian(x, mu, sigma)

def gaussian_mix_dx(x, mu, sigma, w1=0.5, w2=0.5):
	return w1 * gaussian_dx(x, mu, sigma) + w2 * gaussian_dx(x, - mu, sigma)

def V_dx(x, mu, sigma):
	return - gaussian_dx(x, mu, sigma) / gaussian(x, mu, sigma)

def V_dx_mix(x, mu, sigma, w1=0.5, w2=0.5):
	return - gaussian_mix_dx(x, mu, sigma, w1, w2) / gaussian_mix(x, mu, sigma, w1, w2)


def V_linear(x, gamma=1.):
	return gamma * x ** 2

def V_linear_dx(x, gamma=1.):
	return 2 * gamma * x

def mu_linear(x, gamma=1.):
	return jnp.exp(- V_linear(x, gamma))


def high_order_runge_kutta(x, alpha=0.1, h=1e-1, T=100):

	x_list = []
	
	sigma = jnp.sqrt(2 * alpha)
	# sigma = 0.
	# V_tilde = lambda x: alpha * V_dx(x, 0, 1)
	V_tilde = lambda x: alpha * V_dx_mix(x, 3, 1, 0.3, 0.7)
	f = lambda x: - V_tilde(x)

	# sigma = 0.
	# f = lambda x: - V_linear_dx(x)

	def runge_kutta_step(x, t, h):
		csi = jax.random.normal(jax.random.PRNGKey(t * 100), (1,))
		y1 = x + jnp.sqrt(2 * h) * sigma * csi
		y2 = x - 3/8 * h * f(y1) + jnp.sqrt(2 * h) * sigma * csi / 4
		x = x - 1/3 * h * f(y1) + 4/3 * h * f(y2) + sigma * jnp.sqrt(h) * csi
		return x
	
	def runge_kutta_step_2(x, t, h):
		csi = jax.random.normal(jax.random.PRNGKey(t * 100), (1,))
		y1 = x + sigma * jnp.sqrt(h) * csi
		y2 = x - h/2 * f(y1) + sigma/2 * jnp.sqrt(h) * csi
		y3 = x + 3 * h * f(y1) - 2 * h * f(y2) + sigma * jnp.sqrt(h) * csi
		x = x - 3/2 * h * f(y1) + 2 * h * f(y2) + h/2 * f(y3) + sigma * jnp.sqrt(h) * csi
		return x

	for t in range(T):
		x_list.append(x)
		x = runge_kutta_step(x, t, h)
		# x = runge_kutta_step_2(x, h)
	
	return x_list


def main():
	
	T = 300
	N = 100
	x = jax.random.normal(jax.random.PRNGKey(0), (N,))
	x_list = high_order_runge_kutta(x, T=T)

	# x_plot = jnp.linspace(-10, 10, 1000)
	# y = gaussian(x_plot, 0, 1)
	# y /= y.sum()
	# plt.plot(x_plot, y, label='True')
	# plt.plot(x, jnp.zeros_like(x), 'ro', label='Initial')
	# plt.plot(x_new, jnp.zeros_like(x_new), 'bo', label='Final')
	# plt.legend()
	# plt.show()

	# x_plot = jnp.linspace(-10, 10, 1000)
	# y = mu_linear(x_plot)
	# y /= y.sum()
	# plt.plot(x_plot, y, label='True')
	# plt.plot(x, jnp.zeros_like(x), 'ro', label='Initial')
	# plt.plot(x_new, jnp.zeros_like(x_new), 'bo', label='Final')
	# plt.legend()
	# plt.show()

	x_plot = jnp.linspace(-10, 10, 1000)
	fig, ax = plt.subplots()
	for idx, x in zip(range(T), x_list):
		# ax.plot(x_plot, mu_linear(x_plot), label='PDF')
		# ax.plot(x_plot, gaussian(x_plot, 0, 1), label='PDF')
		ax.plot(x_plot, gaussian_mix(x_plot, 3, 1, 0.3, 0.7), label='PDF')
		ax.scatter(x, jnp.zeros(N), alpha=0.1)
		plt.title('iter: {}'.format(idx))
		plt.legend()
		plt.ion()
		plt.draw()
		plt.show()
		plt.pause(0.01)
		ax.clear()


if __name__ == '__main__':
	main()