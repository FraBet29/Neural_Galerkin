import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib import colormaps
from IPython.display import display, clear_output
import time


def plot_function(fn, problem_data, title='Function'):
	x_plot = jnp.linspace(problem_data.domain[0], problem_data.domain[1], problem_data.N)
	plt.plot(x_plot, fn(x_plot))
	plt.xlabel('x')
	plt.ylabel('u')
	plt.title(title)
	plt.show()


def plot_solution(solution, title='Solution'):
	space_time_solution = jnp.array(solution) # (time, space)
	fig, ax = plt.subplots()
	plt.imshow(space_time_solution, interpolation='nearest', cmap=colormaps['coolwarm'])
	plt.colorbar(label='u')
	ax.set_xlabel('x')
	ax.set_ylabel('t')
	ax.set_aspect(int(space_time_solution.shape[1] / space_time_solution.shape[0]))
	plt.title(title)
	plt.show()


def plot_animation(solution, problem_data):
	x_plot = jnp.linspace(problem_data.domain[0], problem_data.domain[1], problem_data.N)
	space_time_solution = jnp.array(solution) # (time, space)
	for i in range(len(solution)):
		plot = plt.plot(x_plot, space_time_solution[i])
		plt.xlim([problem_data.domain[0], problem_data.domain[1]])
		plt.ylim([jnp.min(space_time_solution) - 0.1, jnp.max(space_time_solution) + 0.1])
		display(plot)
		time.sleep(0.001)
		clear_output(wait=True)
		plt.show()


def plot_error(errors, problem_data, title='Error'):
	t_plot = jnp.linspace(0, problem_data.T, int(problem_data.T / problem_data.dt) + 1)
	errors = jnp.array(errors)
	plt.semilogy(t_plot, errors, linestyle='-')
	plt.axhline(y=1, color='r', linestyle='-')
	plt.xlabel('t')
	plt.ylim([1e-3, 1e2])
	plt.title(title)
	plt.grid()
	plt.show()