import jax
import jax.numpy as jnp
import optax
import scipy
from Sampler import *


# def solver(name, *args):
#     '''
#     ODE solver.
#     '''
#     if name == 'rk45':
#         return runge_kutta_scheme(*args)
#     else:
#         raise ValueError(f'Unknown solver: {name}.')


def runge_kutta_scheme(theta_flat, problem_data, n, U, M_fn, F_fn, r_fn=None, sampler='uniform', x_init=None):

    # Sample points in the spatial domain
    if sampler == 'uniform':
        x = uniform_sampling(problem_data, n)
    elif sampler == 'adaptive':
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

    while scheme.t < problem_data.T:

        print(f'  t = {scheme.t:.5f}')

        # Sample points in the spatial domain
        if sampler == 'uniform':
            x = uniform_sampling(problem_data, n, int(scheme.t * 1e6))
        elif sampler == 'adaptive':
            x = adaptive_sampling(theta_flat, problem_data, n, x, scheme.t, M_fn, F_fn, r_fn)
        else:
            raise ValueError(f'Unknown sample mode: {sampler}.')
        
        # Integration step
        scheme.step()
        timesteps.append(scheme.t)

        # Save current solution
        u = U(scheme.y, x_plot)
        solution.append(u)
        
    return solution, timesteps
