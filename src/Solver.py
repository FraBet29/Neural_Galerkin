import jax
import jax.numpy as jnp
import optax
import scipy
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import time
from Sampler import *
from Assemble import *
from Diagnostic import *
from Utils import *


# def solver(name, *args):
#     '''
#     ODE solver.
#     '''
#     if name == 'rk45':
#         return runge_kutta_scheme(*args)
#     else:
#         raise ValueError(f'Unknown solver: {name}.')


def runge_kutta_scheme(theta_flat, problem_data, n, u_fn, rhs, x_init=None, sampler='uniform', reg=0.0, diagnostic_on=False, plot_on=True, seed=0):

    key = jax.random.key(seed)

    # Sample points in the spatial domain
    if sampler == 'uniform':
        x, key = uniform_sampling(problem_data, n, key)
    elif sampler == 'svgd' or sampler == 'svgd_corrected' or sampler == 'high_order_rk':
        if x_init is None:
            raise ValueError('Initial points must be provided for adaptive sampling.')
        x = x_init
    elif sampler == 'weighted':
        store = Orth()
        x, key = uniform_sampling(problem_data, n, key)
        _, w_fn, store = weighted_sampling(u_fn, theta_flat, problem_data, x, store) # better to improve this
    else:
        raise ValueError(f'Unknown sampler: {sampler}.')
    
    if diagnostic_on:
        diagnostic = Diagnostic()
    else:
        diagnostic = None
    
    def rhs_rk45(t, theta_flat):
        if sampler == 'weighted':
            return assemble_weighted(u_fn, rhs, theta_flat, x, t, w_fn, problem_data, store)
        else:
            # Normal equations
            return jnp.linalg.lstsq(M_fn(u_fn, theta_flat, x, reg), F_fn(u_fn, rhs, theta_flat, x, t))[0] # (x, resid, rank, s)
            # Least squares problem
            # return jnp.linalg.lstsq(J_fn(u_fn, theta_flat, x), rhs(theta_flat, x, t, u_fn))[0]

    # Points for plotting and error evaluation
    x_plot = jnp.linspace(problem_data.domain[0], problem_data.domain[1], problem_data.N).reshape(-1, 1)

    # Define the neural network acting on flattened parameters
    U = jax.vmap(u_fn, (None, 0)) # jax.vmap(fun, in_axes)
    
    solution = []
    timesteps = []

    scheme = scipy.integrate.RK45(rhs_rk45, 0, theta_flat, problem_data.T, max_step=problem_data.dt, rtol=1e-4)

    it = 0 # number of iterations

    while scheme.t < problem_data.T:

        # print(f'  t = {scheme.t:.5f}')

        # Save current solution and time
        u = U(scheme.y, x_plot.reshape(-1, 1))
        solution.append(u)
        timesteps.append(scheme.t)

        # Sample points in the spatial domain
        if sampler == 'uniform':
            x, key = uniform_sampling(problem_data, n, key)
        elif sampler == 'svgd':
            x = adaptive_sampling(u_fn, rhs, scheme.y, problem_data, x, scheme.t, gamma=0.25, epsilon=0.05, steps=250,
                                  diagnostic_on=diagnostic_on)
            # x = adaptive_sampling(u_fn, rhs, scheme.y, problem_data, x, scheme.t, gamma=0.25, epsilon=0.01, steps=250,
            #                       diagnostic_on=diagnostic_on)
            # x = adaptive_sampling(u_fn, rhs, scheme.y, problem_data, x, scheme.t, gamma=0.25, epsilon=1e-3, steps=250,
            #                       diagnostic_on=diagnostic_on)
        elif sampler == 'svgd_corrected':
            x = adaptive_sampling(u_fn, rhs, scheme.y, problem_data, x, scheme.t, gamma=0.25, epsilon=0.05, steps=250,
                                  corrected=True, diagnostic_on=diagnostic_on)
            # x = adaptive_sampling(u_fn, rhs, scheme.y, problem_data, x, scheme.t, gamma=0.25, epsilon=0.01, steps=100,
            #                       corrected=True, diagnostic_on=diagnostic_on)
        elif sampler == 'high_order_rk':
            x = high_order_sampler(u_fn, rhs, scheme.y, problem_data, x, scheme.t, gamma=0.25, h=0.05, steps=250, 
                                   diagnostic_on=diagnostic_on)
            # x = high_order_sampler(u_fn, rhs, scheme.y, problem_data, x, scheme.t, gamma=0.25, h=0.1, steps=100, 
            #                        diagnostic_on=diagnostic_on)
        elif sampler == 'weighted':
            store.reset() # need to reset every time theta_flat is updated
            x, w_fn, store = weighted_sampling(u_fn, scheme.y, problem_data, x, store, gamma=0.25, epsilon=0.05, steps=100)
        else:
            raise ValueError(f'Unknown sample mode: {sampler}.')
        
        # Compute the conditioning number and the eigenvalues of the M matrix
        if diagnostic_on:
            M = M_fn(u_fn, scheme.y, x, reg)
            diagnostic.cond.append(jnp.linalg.cond(M))
            eigs = jnp.linalg.eigvals(M)
            diagnostic.max_eig.append(jnp.max(eigs))
            diagnostic.min_eig.append(jnp.min(eigs))
        
        # Integration step
        scheme.step()

        if plot_on and it % 10 == 0:

            plot1 = plt.scatter(x, - 0.1 * jnp.ones(n), color='blue', marker='.')
            if problem_data.exact_sol is not None:
                ref_sol = problem_data.exact_sol(x_plot, scheme.t)
                plot2 = plt.plot(x_plot, ref_sol, '--')
            plot3 = plt.plot(x_plot, U(scheme.y, x_plot.reshape(-1, 1)))
            # delta_theta_flat = predictor_corrector(u_fn, rhs, scheme.y, x, scheme.t)
            # plot4 = plt.plot(x_plot, jnp.abs(r_fn(u_fn, rhs, scheme.y, delta_theta_flat, x_plot.reshape(-1, 1), scheme.t)))
            plt.xlim([problem_data.domain[0], problem_data.domain[1]])
            if problem_data.exact_sol is not None:
                plt.ylim([jnp.min(ref_sol) - 0.5, jnp.max(ref_sol) + 0.5])
            display(plot1)
            if problem_data.exact_sol is not None:
                display(plot2)
            display(plot3)
            # display(plot4)
            time.sleep(0.001)
            clear_output(wait=True)
            if problem_data.exact_sol is not None:
                # plt.legend(['particles', 'exact', 'approx'])
                plt.legend(['particles', 'exact', 'approx', 'residual'])
            else:
                # plt.legend(['particles', 'approx'])
                plt.legend(['particles', 'approx', 'residual'])
            plt.xlabel('x')
            plt.title(f't = {scheme.t:.5f}')
            plt.show()

        it += 1

        if diagnostic_on:
            diagnostic.eigs_final = eigs

    return solution, timesteps, diagnostic


def neural_galerkin(theta, net, problem_data, n, rhs, x_init, sampler='uniform', scheme='rk45', reg=0.0, diagnostic_on=False, plot_on=True, seed=0):

    theta_flat, unravel = jax.flatten_util.ravel_pytree(theta) # flatten a pytree of arrays down to a 1D array
    u_fn = unraveler(net.apply, unravel) # auxiliary function that allows to evaluate the NN starting from the flattened parameters

    print('Run time evolution...')

    start = time.time()

    if scheme == 'rk45':
        solution, timesteps, diagnostic = runge_kutta_scheme(theta_flat, problem_data, n, u_fn, rhs, x_init, sampler, reg, diagnostic_on, plot_on, seed)
    elif scheme == 'bwe':
        # solution, timesteps, diagnostic = backward_euler_scheme(theta_flat, problem_data, n, u_fn, rhs, x_init, sampler, diagnostic_on, plot_on)
        raise NotImplementedError('Backward Euler scheme is not implemented yet.')
    else:
        raise ValueError(f'Unknown scheme: {scheme}.')
    
    print(f'Elapsed time: {time.time() - start:.3f} s')

    return solution, timesteps, diagnostic