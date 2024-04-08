import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib import colormaps
from IPython.display import display, clear_output
from scipy.interpolate import interp1d
import time


def plot_function(fn, problem_data, title='Function'):
	x_plot = jnp.linspace(problem_data.domain[0], problem_data.domain[1], problem_data.N)
	plt.plot(x_plot, fn(x_plot))
	plt.xlabel('x')
	plt.ylabel('u')
	plt.title(title)
	plt.show()


def plot_solution(solution, timesteps, problem_data, title='Solution'):
	
	space_time_solution = jnp.array(solution) # (time, space)
	interpolator = interp1d(timesteps, space_time_solution, axis=0, kind='linear', fill_value="extrapolate")
	t_plot = jnp.linspace(0, problem_data.T, 200)
	space_time_solution = interpolator(t_plot)

	fig, ax = plt.subplots()
	plt.imshow(space_time_solution.T, interpolation='nearest', cmap=colormaps['coolwarm'], origin='lower', 
               extent=[0, problem_data.T, problem_data.domain[0], problem_data.domain[1]], aspect='auto')
	plt.colorbar(label='u')
	ax.set_xlabel('t')
	ax.set_ylabel('x')
	plt.title(title)
	plt.show()


def plot_animation(solution, timesteps, problem_data):

	x_plot = jnp.linspace(problem_data.domain[0], problem_data.domain[1], problem_data.N)
	space_time_solution = jnp.array(solution) # (time, space)
	interpolator = interp1d(timesteps, space_time_solution, axis=0, kind='linear', fill_value="extrapolate")
	t_plot = jnp.linspace(0, problem_data.T, 200)
	space_time_solution = interpolator(t_plot)

	for i in range(space_time_solution.shape[0]):
		plot = plt.plot(x_plot, space_time_solution[i])
		plt.xlim([problem_data.domain[0], problem_data.domain[1]])
		plt.ylim([jnp.min(space_time_solution) - 0.1, jnp.max(space_time_solution) + 0.1])
		display(plot)
		time.sleep(0.001)
		clear_output(wait=True)
		plt.show()


def plot_error(errors, timesteps, title='Error'):
	t_plot = jnp.array(timesteps)
	errors = jnp.array(errors)
	plt.semilogy(t_plot, errors, linestyle='-')
	plt.axhline(y=1, color='r', linestyle='-')
	plt.xlabel('t')
	plt.ylim([min(1e-3, 1e-1 * jnp.min(errors)), max(1e1, 1e1 * jnp.max(errors))])
	plt.title(title)
	plt.grid()
	plt.show()


def plot_error_comparison(errors_list, timesteps_list, names_list, title='Error comparison'):
	assert(len(errors_list) == len(timesteps_list) == len(names_list))
	for errors, timesteps in zip(errors_list, timesteps_list):
		t_plot = jnp.array(timesteps)
		errors = jnp.array(errors)
		plt.semilogy(t_plot, errors, linestyle='-')
	plt.axhline(y=1, color='r', linestyle='-')
	plt.xlabel('t')
	plt.ylim([min(1e-3, 1e1 * jnp.min(errors)), max(1e2, 1e1 * jnp.max(errors))])
	plt.legend(names_list)
	plt.title(title)
	plt.grid()
	plt.show()