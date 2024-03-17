import jax
import jax.numpy as jnp


# Function to evaluate the neural network on the unflattened parameters
def unraveler(f, unravel, axis=0):
    '''
    Function taken from: https://github.com/julesberman/RSNG/blob/main/rsng/dnn.py
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
        return f(*args, **kwargs) # evaluate the neural network on the unflattened parameters
    return wrapper


# Take gradient and then squeeze
def gradsqz(f, *args, **kwargs):
    '''
    Function taken from: https://github.com/julesberman/RSNG/blob/main/allen_cahn.ipynb
    '''
    return lambda *fargs, **fkwargs: jnp.squeeze(jax.grad(f, *args, **kwargs)(*fargs, **fkwargs))
