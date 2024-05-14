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


def runge_kutta_scheme(theta_flat, problem_data, n, u_fn, rhs, x_init=None, sampler='uniform', diagnostic_on=False, plot_on=True):

    # Sample points in the spatial domain
    if sampler == 'uniform':
        x = uniform_sampling(problem_data, n)
    elif sampler == 'svgd' or sampler == 'svgd_corrected':
        if x_init is None:
            raise ValueError('Initial points must be provided for adaptive sampling.')
        x = x_init
    elif sampler == 'weighted':
        x = uniform_sampling(problem_data, n)
        w_fn = lambda y: jnp.array([1]) # initial weights all equal to 1
    else:
        raise ValueError(f'Unknown sampler: {sampler}.')
    
    def rhs_rk45(t, theta_flat):
        if sampler == 'weighted':
            M, F = assemble_weighted(u_fn, rhs, theta_flat, x, t, w_fn)
            return jnp.linalg.lstsq(M, F)[0]
            # return jnp.linalg.lstsq(M_fn_weighted(u_fn, theta_flat, x, w_fn), F_fn_weighted(u_fn, rhs, theta_flat, x, t, w_fn))[0]
        else:
            return jnp.linalg.lstsq(M_fn(u_fn, theta_flat, x), F_fn(u_fn, rhs, theta_flat, x, t))[0]

    # Points for plotting and error evaluation
    x_plot = jnp.linspace(problem_data.domain[0], problem_data.domain[1], problem_data.N).reshape(-1, 1)

    # Define the neural network acting on flattened parameters
    U = jax.vmap(u_fn, (None, 0)) # jax.vmap(fun, in_axes)
    
    solution = []
    timesteps = []
    if diagnostic_on:
        diagnostic = Diagnostic()
    else:
        diagnostic = None

    scheme = scipy.integrate.RK45(rhs_rk45, 0, theta_flat, problem_data.T, max_step=problem_data.dt, rtol=1e-4)

    it = 0 # number of iterations

    while scheme.t < problem_data.T:

        if sampler == 'weighted':
            # M = M_fn_weighted(u_fn, scheme.y, x, w_fn)
            M, _ = assemble_weighted(u_fn, rhs, scheme.y, x, scheme.t, w_fn)
            print('cond(M) =', jnp.linalg.cond(M))

        # print(f'  t = {scheme.t:.5f}')

        # Save current solution and time
        u = U(scheme.y, x_plot.reshape(-1, 1))
        solution.append(u)
        timesteps.append(scheme.t)

        # Sample points in the spatial domain
        if sampler == 'uniform':
            x = uniform_sampling(problem_data, n, int(scheme.t * 1e6))
        elif sampler == 'svgd':
            x = adaptive_sampling(u_fn, rhs, scheme.y, problem_data, x, scheme.t, gamma=0.25, epsilon=0.05, steps=250,
                                  diagnostic_on=diagnostic_on)
        elif sampler == 'svgd_corrected':
            x = adaptive_sampling(u_fn, rhs, scheme.y, problem_data, x, scheme.t, gamma=0.25, epsilon=0.05, steps=250,
                                  corrected=True, diagnostic_on=diagnostic_on)
        elif sampler == 'weighted':
            x, w_fn = weighted_sampling(u_fn, scheme.y, problem_data, x, gamma=0.25, epsilon=0.05, steps=250)
        else:
            raise ValueError(f'Unknown sample mode: {sampler}.')
        
        # Compute the conditioning number and the eigenvalues of the M matrix
        if diagnostic_on:
            M = M_fn(u_fn, scheme.y, x)
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
            # time.sleep(0.001)
            clear_output(wait=True)
            
            if problem_data.exact_sol is not None:
                plt.legend(['particles', 'exact', 'approx'])
                # plt.legend(['particles', 'exact', 'approx', 'residual'])
            else:
                plt.legend(['particles', 'approx'])
                # plt.legend(['particles', 'approx', 'residual'])
            plt.title(f't = {scheme.t:.5f}')
            plt.show()

        it += 1

        if diagnostic_on:
            diagnostic.eigs_final = eigs

    return solution, timesteps, diagnostic


# @partial(jax.jit, static_argnums=(0, 1, ))
# def implicit_integrator(u_fn, rhs, theta_flat, x, t):
    
#     theta_flat_k = jnp.copy(theta_flat)
    
#     optimizer = optax.adam(1e-3)
#     opt_state = optimizer.init(theta_flat)
    
#     for i in range(100):
#         # if i % 10 == 0:
#         #     print(f'ADAM iter: {i}/100')
#         #     print(r_loss(u_fn, rhs, theta_flat, theta_flat - theta_flat_k, x, t))
#         grads = jax.grad(r_loss, argnums=2)(u_fn, rhs, theta_flat, theta_flat - theta_flat_k, x, t)
#         updates, opt_state = optimizer.update(grads, opt_state)
#         theta_flat = optax.apply_updates(theta_flat, updates)

#     r = r_loss(u_fn, rhs, theta_flat, theta_flat - theta_flat_k, x, t) # final residual

#     return theta_flat, r


# def backward_euler_scheme(theta_flat, problem_data, n, u_fn, rhs, x_init=None, sampler='uniform', diagnostic_on=False, plot_on=True):

#     if sampler == 'svgd':
#         if x_init is None:
#             raise ValueError('Initial points must be provided for adaptive sampling.')
#         x = x_init

#     # Points for plotting and error evaluation
#     x_plot = jnp.linspace(problem_data.domain[0], problem_data.domain[1], problem_data.N).reshape(-1, 1)

#     # Define the neural network acting on flattened parameters
#     U = jax.vmap(u_fn, (None, 0)) # jax.vmap(fun, in_axes)
    
#     solution = []
#     timesteps = []
#     if diagnostic_on:
#         diagnostic = Diagnostic()
#     else:
#         diagnostic = None

#     it = 0 # number of iterations
#     t = 0

#     while t < problem_data.T:

#         # print(f'  t = {t:.5f}')

#         # Save current solution and time
#         u = U(theta_flat, x_plot.reshape(-1, 1))
#         solution.append(u)
#         timesteps.append(t)

#         # Sample points in the spatial domain
#         if sampler == 'uniform':
#             x = uniform_sampling(problem_data, n, int(t * 1e6))
#         elif sampler == 'svgd':
#             x = adaptive_sampling(u_fn, rhs, theta_flat, problem_data, x, t, gamma=0.25, epsilon=0.05, steps=250,
#                                   diagnostic_on=diagnostic_on)
#         else:
#             raise ValueError(f'Unknown sample mode: {sampler}.')
        
#         # Compute the conditioning number and the eigenvalues of the M matrix
#         if diagnostic_on:
#             M = M_fn(u_fn, theta_flat, x)
#             diagnostic.cond.append(jnp.linalg.cond(M))
#             eigs = jnp.linalg.eigvals(M)
#             diagnostic.max_eig.append(jnp.max(eigs))
#             diagnostic.min_eig.append(jnp.min(eigs))
        
#         # Integration step with ADAM
#         # theta_flat_k = jnp.copy(theta_flat)
#         # optimizer = optax.adam(1e-3)
#         # opt_state = optimizer.init(theta_flat)
#         # for i in range(100):
#         #     # if i % 10 == 0:
#         #     #     print(f'ADAM iter: {i}/100')
#         #     #     print(r_loss(u_fn, rhs, theta_flat, theta_flat - theta_flat_k, x, t))
#         #     grads = jax.grad(r_loss, argnums=2)(u_fn, rhs, theta_flat, theta_flat - theta_flat_k, x, t)
#         #     updates, opt_state = optimizer.update(grads, opt_state)
#         #     theta_flat = optax.apply_updates(theta_flat, updates)
            
#         theta_flat, r = implicit_integrator(u_fn, rhs, theta_flat, x, t)

#         if int(t / problem_data.dt) % 10 == 0:
#             # print(f'ADAM residual: {r_loss(u_fn, rhs, theta_flat, theta_flat - theta_flat_k, x, t):.3f}')
#             print(f'ADAM residual: {r:.3f}')

#         if plot_on and it % 10 == 0:

#             plot1 = plt.scatter(x, - 0.1 * jnp.ones(n), color='blue', marker='.')
#             if problem_data.exact_sol is not None:
#                 ref_sol = problem_data.exact_sol(x_plot, t)
#                 plot2 = plt.plot(x_plot, ref_sol, '--')
#             plot3 = plt.plot(x_plot, U(theta_flat, x_plot.reshape(-1, 1)))
#             plt.xlim([problem_data.domain[0], problem_data.domain[1]])
#             if problem_data.exact_sol is not None:
#                 plt.ylim([jnp.min(ref_sol) - 0.5, jnp.max(ref_sol) + 0.5])
#             display(plot1)
#             if problem_data.exact_sol is not None:
#                 display(plot2)
#             display(plot3)
#             # time.sleep(0.001)
#             clear_output(wait=True)
            
#             if problem_data.exact_sol is not None:
#                 plt.legend(['particles', 'exact', 'approx'])
#             else:
#                 plt.legend(['particles', 'approx'])
#             plt.title(f't = {t:.5f}')
#             plt.show()

#         it += 1
#         t += problem_data.dt

#     if diagnostic_on:
#         diagnostic.eigs_final = eigs

#     return solution, timesteps, diagnostic


def neural_galerkin(theta, net, problem_data, n, rhs, x_init, sampler='uniform', scheme='rk45', diagnostic_on=False, plot_on=True):

    theta_flat, unravel = jax.flatten_util.ravel_pytree(theta) # flatten a pytree of arrays down to a 1D array
    u_fn = unraveler(net.apply, unravel) # auxiliary function that allows to evaluate the NN starting from the flattened parameters

    print('Run time evolution...')

    start = time.time()

    if scheme == 'rk45':
        solution, timesteps, diagnostic = runge_kutta_scheme(theta_flat, problem_data, n, u_fn, rhs, x_init, sampler, diagnostic_on, plot_on)
    elif scheme == 'bwe':
        # solution, timesteps, diagnostic = backward_euler_scheme(theta_flat, problem_data, n, u_fn, rhs, x_init, sampler, diagnostic_on, plot_on)
        raise NotImplementedError('Backward Euler scheme is not implemented yet.')
    else:
        raise ValueError(f'Unknown scheme: {scheme}.')
    
    print(f'Elapsed time: {time.time() - start:.3f} s')

    return solution, timesteps, diagnostic