import jax
import jax.numpy as jnp
from functools import partial


def unraveler(f, unravel, axis=0):
    '''
    Function taken from: https://github.com/julesberman/RSNG/blob/main/rsng/dnn.py
    Evaluate the neural network on the unflattened parameters.
    Args:
        f: callable, network 'apply' method
        unravel: callable, to unflatten an array back into a pytree
        axis: int
    Returns:
        callable, to evaluate the neural network on the unflattened parameters
    '''
    def wrapper(*args, **kwargs):
        val = args[axis]
        if (type(val) != dict): # if not already unflattened
            args = list(args)
            args[axis] = unravel(val)
            args = tuple(args)
        # return f(*args, **kwargs) # evaluate the neural network on the unflattened parameters
        return f(*args, **kwargs).squeeze() # evaluate the neural network on the unflattened parameters, squeeze the output to return a scalar
    return wrapper


def gradsqz(f, *args, **kwargs):
    '''
    Function taken from: https://github.com/julesberman/RSNG/blob/main/allen_cahn.ipynb
    Args:
        f: callable, function to be differentiated
    '''
    return lambda *fargs, **fkwargs: jnp.squeeze(jax.grad(f, *args, **kwargs)(*fargs, **fkwargs))


def compute_error(solution, timesteps, exact_solution, problem_data):

    x_plot = jnp.linspace(problem_data.domain[0], problem_data.domain[1], problem_data.N)
    t_plot = jnp.array(timesteps)
    ref_solution = exact_solution(x_plot, t_plot).T
    space_time_solution = jnp.array(solution) # (time, space)

    diff = ref_solution - space_time_solution
    cumulative_ref_norms = jnp.cumsum(jnp.linalg.norm(ref_solution, axis=1))
    cumulative_diff_norms = jnp.cumsum(jnp.linalg.norm(diff, axis=1))
    errors = cumulative_diff_norms / cumulative_ref_norms
    errors = jnp.nan_to_num(errors, nan=0.0) # handle division by zero
    errors = errors.tolist()

    return errors
