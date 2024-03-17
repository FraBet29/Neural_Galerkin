from Data import *
from NeuralNetwork import *
from InitialFit import *
from JaxUtils import *
# from AssembleSystem import *
import matplotlib.pyplot as plt
from matplotlib import colormaps
import scipy


####### Initialization #######

run_training = False

def plot_exact_initial_condition():
	x_plot = jnp.linspace(problem_data.domain[0], problem_data.domain[1], problem_data.N)
	plt.plot(x_plot, exactKdVTwoSol(x_plot, 0))
	plt.xlabel('x')
	plt.ylabel('u_0')
	plt.title('Exact initial condition')
	plt.show()

# plot_exact_initial_condition()

if run_training:
	
	# Define the neural network
	shallow_net = ShallowNet(problem_data.d, training_data.m, L)
	
    # Fit the initial condition
	theta_init = init_neural_galerkin(shallow_net)

	# Save the initial parameters
	jnp.save('./data/theta_init_' + problem_data.name + '.npy', theta_init)


####### Time evolution #######

shallow_net = ShallowNet(problem_data.d, training_data.m, L)

# Reload the initial parameters
theta = jnp.load('./data/theta_init_' + problem_data.name + '.npy', allow_pickle=True).item()

# Plot initial fit
x_plot = jnp.linspace(problem_data.domain[0], problem_data.domain[1], problem_data.N)
u_pred = shallow_net.apply(theta, x_plot.reshape(-1, 1))
plt.plot(x_plot, exactKdVTwoSol(x_plot, 0), label='True')
plt.plot(x_plot, u_pred, label='Fitted')
plt.title('Initial fit')
plt.legend()
plt.show()


# Update parameters
theta_flat, unravel = jax.flatten_util.ravel_pytree(theta) # flatten a pytree of arrays down to a 1D array
u_scalar = unraveler(shallow_net.apply, unravel)

# Define gradients
U = jax.vmap(u_scalar, (None, 0)) # jax.vmap(fun, in_axes)
U_dtheta = jax.vmap(jax.grad(u_scalar), (None, 0))
U_dx = jax.vmap(gradsqz(u_scalar, 1), (None, 0))
U_dddx = jax.vmap(gradsqz(gradsqz(gradsqz(u_scalar, 1), 1), 1), (None, 0))

# Source term for the KdV equation
def rhs(theta_flat, x, t):
    u = U(theta_flat, x)
    u_x = U_dx(theta_flat, x)
    u_xxx = U_dddx(theta_flat, x)
    return - u_xxx - 6 * u * u_x

@jax.jit
def assemble_system(theta_flat, x, t):
    # theta_flat = jax.flatten_util.ravel_pytree(theta)[0]
    u_dth = U_dtheta(theta_flat, x)
    M = jnp.mean(u_dth[:, :, jnp.newaxis] * u_dth[:, jnp.newaxis, :], axis=0)
    f = rhs(theta_flat, x, t) # source term
    F = jnp.mean(u_dth * f[:, jnp.newaxis], axis=0)
    return M, F

@jax.jit
def M_fn(theta_flat, x):
    # theta_flat = jax.flatten_util.ravel_pytree(theta)[0]
    u_dth = U_dtheta(theta_flat, x)
    M = jnp.mean(u_dth[:, :, jnp.newaxis] * u_dth[:, jnp.newaxis, :], axis=0)
    return M

@jax.jit
def F_fn(theta_flat, x, t):
    # theta_flat = jax.flatten_util.ravel_pytree(theta)[0]
    u_dth = U_dtheta(theta_flat, x)
    f = rhs(theta_flat, x, t) # source term
    F = jnp.mean(u_dth[:, :] * f[:, jnp.newaxis], axis=0)
    return F

@jax.jit
def r_fn(theta_flat, theta_flat_k, x, t):
    return jnp.dot(M_fn(theta_flat, x), theta_flat) - jnp.dot(M_fn(theta_flat, x), theta_flat_k) - \
        problem_data.dt * F_fn(theta_flat, x, t + problem_data.dt)

@jax.jit
def r_loss(theta_flat, theta_flat_k, x, t):
    return jnp.linalg.norm(r_fn(theta_flat, theta_flat_k, x, t))

@jax.jit
def rhs_RK45(theta_flat, x, t):
    return jnp.linalg.lstsq(M_fn(theta_flat, x), F_fn(theta_flat, x, t))[0]


def neural_galerkin(net, theta):

    x_plot = jnp.linspace(problem_data.domain[0], problem_data.domain[1], problem_data.N) # for plotting and error evaluation

    # Instantiate a class to assemble the system of equations
    # system = AssembleSystem(net, theta_init)

    solution = []

    t = 0
    print('Run time evolution...')
    while t < problem_data.T:

        if int(t / problem_data.dt) % 50 == 0:
            print(f'  t = {t:.2f}')

        # Update parameters
        theta_flat, unravel = jax.flatten_util.ravel_pytree(theta) # flatten a pytree of arrays down to a 1D array
        u_scalar = unraveler(net.apply, unravel)

        # Define gradients
        U = jax.vmap(u_scalar, (None, 0)) # jax.vmap(fun, in_axes)
        U_dtheta = jax.vmap(jax.grad(u_scalar), (None, 0))
        U_dx = jax.vmap(gradsqz(u_scalar, 1), (None, 0))
        U_dddx = jax.vmap(gradsqz(gradsqz(gradsqz(u_scalar, 1), 1), 1), (None, 0))
        
        # Sample points in the spatial domain
        x = jax.random.uniform(jax.random.key(int(t * 1e4)), (problem_data.n, problem_data.d), 
                               minval=problem_data.domain[0], maxval=problem_data.domain[1]).reshape(-1, 1)

        # Save current solution
        u = U(theta, x_plot.reshape(-1, 1))
        solution.append(u)

        # Approximate M and F
        # M, F = assemble_system(theta, x, t)

        # Find theta dot
        # theta_dot_flat = jnp.linalg.solve(M, F)

        # RK45
        theta_flat = jax.flatten_util.ravel_pytree(theta)[0]
        def rhs_RK45_wrapper(t, theta_flat):
            return rhs_RK45(theta_flat, x, t)
        theta_flat_k = jnp.copy(theta_flat)
        RK45 = scipy.integrate.RK45(rhs_RK45_wrapper, t, theta_flat_k, t + problem_data.dt)
        RK45.step()
        theta_flat = RK45.y

        # Implicit Euler with ADAM
        # theta_flat = jax.flatten_util.ravel_pytree(theta)[0]
        # theta_flat_k = jnp.copy(theta_flat)
        # optimizer = optax.adam(1e-3)
        # opt_state = optimizer.init(theta_flat)
        # for _ in range(50):
        #     # if i % 10 == 0:
        #     #     print(f'Adam iter: {i}/100')
        #     #     print(r_loss(theta_flat, theta_flat_k, x, t))
        #     grads = jax.grad(r_loss)(theta_flat, theta_flat_k, x, t)
        #     updates, opt_state = optimizer.update(grads, opt_state)
        #     theta_flat = optax.apply_updates(theta_flat, updates)

        # if int(t / problem_data.dt) % 10 == 0:
        #     print(f' t = {t:.2f}, ADAM residual: {r_loss(theta_flat, theta_flat_k, x, t):.5f}')

        theta = unravel(theta_flat)
        t = t + problem_data.dt

    return solution


# Run the neural Galerkin method
solution = neural_galerkin(shallow_net, theta)

# Plot the solution
space_time_solution = jnp.array(solution) # (time, space)
fig, ax = plt.subplots()
plt.imshow(space_time_solution, interpolation='nearest', cmap=colormaps['coolwarm'])
plt.colorbar(label='u')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_aspect(int(space_time_solution.shape[1] / space_time_solution.shape[0]))
plt.title('Solution')
plt.show()


####### Compute error #######

def compute_error(solution):

    x_plot = jnp.linspace(problem_data.domain[0], problem_data.domain[1], problem_data.N)
    t_plot = jnp.linspace(0, problem_data.T, int(problem_data.T / problem_data.dt) + 1)
    ref_solution = exactKdVTwoSol(x_plot, t_plot).T
    space_time_solution = jnp.array(solution) # (time, space)

    diff = ref_solution - space_time_solution
    cumulative_ref_norms = jnp.cumsum(jnp.linalg.norm(ref_solution, axis=1))
    cumulative_diff_norms = jnp.cumsum(jnp.linalg.norm(diff, axis=1))
    errors = cumulative_diff_norms / cumulative_ref_norms
    errors = jnp.nan_to_num(errors, nan=0.0) # handle division by zero
    errors = errors.tolist()

    return errors


errors = compute_error(solution)
errors_plot = jnp.array(errors)

t_plot = jnp.linspace(0, problem_data.T, int(problem_data.T / problem_data.dt) + 1)
plt.semilogy(t_plot, errors_plot, linestyle='-')
plt.axhline(y=1, color='r', linestyle='-')
plt.xlabel('t')
plt.ylim([1e-3, 1e2])
plt.title('Relative L2 error')
plt.grid()
plt.show()