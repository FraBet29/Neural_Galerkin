import jax
import jax.numpy as jnp
import optax
import scipy
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import time
from Sampler import *


# def solver(name, *args):
#     '''
#     ODE solver.
#     '''
#     if name == 'rk45':
#         return runge_kutta_scheme(*args)
#     else:
#         raise ValueError(f'Unknown solver: {name}.')


def runge_kutta_scheme(theta_flat, problem_data, n, U, M_fn, F_fn, r_fn=None, sampler='uniform', x_init=None, plot_on=True):

    # Sample points in the spatial domain
    if sampler == 'uniform':
        if x_init is None:
            x = uniform_sampling(problem_data, n)
        else:
            x = x_init
    elif sampler == 'svgd':
        if x_init is None:
            raise ValueError('Initial points must be provided for adaptive sampling.')
        x = x_init
    else:
        raise ValueError(f'Unknown sampler: {sampler}.')
    
    def rhs_RK45(t, theta_flat):
        return jnp.linalg.lstsq(M_fn(theta_flat, x), F_fn(theta_flat, x, t))[0]

    # Points for plotting and error evaluation
    x_plot = jnp.linspace(problem_data.domain[0], problem_data.domain[1], problem_data.N).reshape(-1, 1)
    
    solution = []
    timesteps = []

    scheme = scipy.integrate.RK45(rhs_RK45, 0, theta_flat, problem_data.T, max_step=problem_data.dt, rtol=1e-4)

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
            x = adaptive_sampling(scheme.y, problem_data, n, x, scheme.t, M_fn, F_fn, r_fn)
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
            plot1 = plt.scatter(x, jnp.zeros(n), color='blue', marker='.')
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
