import jax
import jax.numpy as jnp
import scipy
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import time
from Sampler import *
from Assemble import *
from Utils import *


# def solver(name, *args):
#     '''
#     ODE solver.
#     '''
#     if name == 'rk45':
#         return runge_kutta_scheme(*args)
#     else:
#         raise ValueError(f'Unknown solver: {name}.')


def runge_kutta_scheme(theta_flat, problem_data, n, u_fn, rhs, x_init=None, sampler='uniform', plot_on=True):

    # Sample points in the spatial domain
    if sampler == 'uniform':
        x = uniform_sampling(problem_data, n)
    elif sampler == 'svgd':
        if x_init is None:
            raise ValueError('Initial points must be provided for adaptive sampling.')
        x = x_init
    else:
        raise ValueError(f'Unknown sampler: {sampler}.')
    
    def rhs_rk45(t, theta_flat):
        return jnp.linalg.lstsq(M_fn(u_fn, theta_flat, x), F_fn(u_fn, rhs, theta_flat, x, t))[0]

    # Points for plotting and error evaluation
    x_plot = jnp.linspace(problem_data.domain[0], problem_data.domain[1], problem_data.N).reshape(-1, 1)

    # Define the neural network acting on flattened parameters
    U = jax.vmap(u_fn, (None, 0)) # jax.vmap(fun, in_axes)
    
    solution = []
    timesteps = []

    scheme = scipy.integrate.RK45(rhs_rk45, 0, theta_flat, problem_data.T, max_step=problem_data.dt, rtol=1e-4)

    it = 0 # number of iterations

    while scheme.t < problem_data.T:

        print(f'  t = {scheme.t:.5f}')

        # Save current solution
        u = U(scheme.y, x_plot.reshape(-1, 1))
        solution.append(u)
        timesteps.append(scheme.t)

        # Sample points in the spatial domain
        if sampler == 'uniform':
            x = uniform_sampling(problem_data, n, int(scheme.t * 1e6))
        elif sampler == 'svgd':
            x = adaptive_sampling(u_fn, rhs, scheme.y, problem_data, x, scheme.t, gamma=0.25, epsilon=0.05, steps=250)
        else:
            raise ValueError(f'Unknown sample mode: {sampler}.')
        
        # Integration step
        scheme.step()
        timesteps.append(scheme.t)

        # Save current solution
        u = U(scheme.y, x_plot)
        solution.append(u)

        if plot_on and it % 10 == 0:
            ref_sol = problem_data.exact_sol(x_plot, scheme.t)
            plot1 = plt.scatter(x, - 0.1 * jnp.ones(n), color='blue', marker='.')
            plot2 = plt.plot(x_plot, ref_sol, '--')
            plot3 = plt.plot(x_plot, U(scheme.y, x_plot.reshape(-1, 1)))
            plt.xlim([problem_data.domain[0], problem_data.domain[1]])
            plt.ylim([jnp.min(ref_sol) - 0.5, jnp.max(ref_sol) + 0.5])
            display(plot1)
            display(plot2)
            display(plot3)
            # time.sleep(0.001)
            clear_output(wait=True)
            
            plt.legend(['particles', 'exact', 'approx'])
            plt.title(f't = {scheme.t:.5f}')
            plt.show()

        it += 1
        
    return solution, timesteps


def neural_galerkin(theta, net, problem_data, n, rhs, x_init, sampler='uniform', scheme='rk45'):

    theta_flat, unravel = jax.flatten_util.ravel_pytree(theta) # flatten a pytree of arrays down to a 1D array
    u_fn = unraveler(net.apply, unravel) # auxiliary function that allows to evaluate the NN starting from the flattened parameters

    print('Run time evolution...')

    if scheme == 'rk45':
        solution, timesteps = runge_kutta_scheme(theta_flat, problem_data, n, u_fn, rhs, x_init, sampler)
    else:
        raise ValueError(f'Unknown scheme: {scheme}.')

    return solution, timesteps