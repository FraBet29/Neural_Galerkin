import numpy as np
from scipy.stats import uniform, norm, gaussian_kde
import matplotlib.pyplot as plt
import time
from svgd import SVGD
from wasserstein import wasserstein_1d


if __name__ == '__main__':
	'''
	x0: initial particles
	dlnprob: returns first order derivative of log probability
	n_iter: number of iterations
	stepsize: initial learning rate 
	'''
	n = 100 # number of particles
	n_iter = 1000 # number of iterations
	stepsize = 0.05 # stepsize

	# set numpy random seed
	np.random.seed(123)


	def gauss(x, mu, sigma):
		return np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))
	

	def first_demo(gauss_mean):
		x0 = norm.rvs(loc=-10, scale=1, size=[n, 1])
		dlnprob = lambda x: - (gauss(x, - gauss_mean, 1) / 3 * (x + gauss_mean) + 2 * gauss(x, gauss_mean, 1) / 3 * (x - gauss_mean)) / (gauss(x, - gauss_mean, 1) / 3 + 2 * gauss(x, gauss_mean, 1) / 3)
		theta, theta_hist, wass_hist = SVGD().update(x0, dlnprob, n_iter, stepsize, alpha=0.9, debug=True)
		return theta, theta_hist, wass_hist
	

	def second_demo(gauss_mean):
		x0 = uniform.rvs(loc=-10, scale=20, size=[n, 1])
		dlnprob = lambda x: - (gauss(x, - gauss_mean, 1) / 3 * (x + gauss_mean) + 2 * gauss(x, gauss_mean, 1) / 3 * (x - gauss_mean)) / (gauss(x, - gauss_mean, 1) / 3 + 2 * gauss(x, gauss_mean, 1) / 3)
		theta, theta_hist, wass_hist = SVGD().update(x0, dlnprob, n_iter, stepsize, alpha=0.9, debug=True)
		return theta, theta_hist, wass_hist
	

	def third_demo(s1, s2):

		def prob_prime(x, s1, s2):
			def f(x, s):
				return np.sin(5 * x - s) * np.cos(0.5 * x - s)
			def f_prime(x, s):
				return (5 * np.cos(5 * x - s) * np.cos(0.5 * x - s)) - (0.5 * np.sin(5 * x - s) * np.sin(0.5 * x - s))
			term1 = 0.7 * f_prime(x, s1) * (f(x, s1) / np.abs(f(x, s1)))
			term2 = 0.3 * f_prime(x, s2) * (f(x, s2) / np.abs(f(x, s2)))
			return term1 * (x > s1 - np.pi) * (x < s1 + np.pi) + term2 * (x > s2 - np.pi) * (x < s2 + np.pi)
		
		dlnprob = lambda x: prob_prime(x, s1, s2)
		# x0 = uniform.rvs(loc=-10, scale=20, size=[n, 1])
		x0 = norm.rvs(loc=0, scale=1, size=[n, 1])
		theta, theta_hist, wass_hist = SVGD().update(x0, dlnprob, n_iter, stepsize, alpha=0.9, debug=True)
		return theta, theta_hist, wass_hist


	# gauss_mean = 2
	# theta, theta_hist, wass_hist = first_demo(gauss_mean)
	
	# gauss_mean = 5
	# theta, theta_hist, wass_hist = second_demo(gauss_mean)
	
	base = lambda x, s: np.abs(np.sin(5 * x - s) * np.cos(0.5 * x - s))
	prob = lambda x, s1, s2: 0.7 * base(x, s1) * (x > s1 - np.pi) * (x < s1 + np.pi) + 0.3 * base(x, s2) * (x > s2 - np.pi) * (x < s2 + np.pi)
	s1, s2 = 0, 2 * np.pi
	theta, theta_hist, wass_hist = third_demo(s1, s2)


	x_plot = np.linspace(-15, 15, 1000)

	# plot last theta
	kernel = gaussian_kde(theta.squeeze())
	plt.plot(x_plot, kernel(x_plot), label='PDF')
	# plt.plot(x_plot, norm.pdf(x_plot, loc=-gauss_mean, scale=1) / 3 + 2 * norm.pdf(x_plot, loc=gauss_mean, scale=1) / 3, label='Target')
	plt.plot(x_plot, prob(x_plot, s1, s2), label='Target')
	plt.xlim([-15, 15])
	plt.legend()
	plt.show()

	# plot wasserstein distance (cumsum)
	plt.plot(np.cumsum(np.array(wass_hist)) / np.arange(1, n_iter + 1))
	# plt.plot(wass_hist)
	plt.xlabel('Iteration')
	plt.ylabel('Wasserstein Distance (cumsum)')
	plt.show()

	# plot wasserstein distance (movmean)
	def movmean(x, w):
		return np.convolve(x, np.ones(w), 'valid') / w
	plt.plot(movmean(np.array(wass_hist), 100))
	plt.xlabel('Iteration')
	plt.ylabel('Wasserstein Distance (movmean)')
	plt.show()

	# print particles evolution
	fig, ax = plt.subplots()
	for idx, theta in enumerate(theta_hist):
		kernel = gaussian_kde(theta.squeeze())
		ax.plot(x_plot, kernel(x_plot), label='PDF')
		# ax.plot(x_plot, norm.pdf(x_plot, loc=-gauss_mean, scale=1) / 3 + 2 * norm.pdf(x_plot, loc=gauss_mean, scale=1) / 3, label='Target')
		ax.plot(x_plot, prob(x_plot, s1, s2), label='Target')
		ax.scatter(theta, np.zeros(n), alpha=0.1)
		plt.legend()
		# plt.ylim([-0.1, 0.5])
		plt.ylim([-0.1, 1])
		plt.xlim([-15, 15])
		plt.title('iter: {}'.format(idx))
		plt.ion()
		plt.draw()
		plt.show()
		plt.pause(0.01)
		ax.clear()

