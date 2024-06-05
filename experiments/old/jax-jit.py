import jax
import jax.numpy as jnp
import time

# Define a simple jittable function
@jax.jit
def f(x):
    return x + 1

# Define a function taking the result of the jitted function as an argument
@jax.jit
def g(result_of_f, x):
    return 2 * result_of_f

n = 10 ** 6
key = jax.random.key(0)
x = jax.random.uniform(key, (n,))

start = time.time()
result_of_f = f(x)
result_of_g = g(result_of_f, 1)
end = time.time()
print(f"Elapsed time: {end - start:.6f} s")

# Compare with the non-jitted version

def f_nonj(x):
    return x + 1

def g_nonj(result_of_f_nonj, x):
    return 2 * result_of_f_nonj

start = time.time()
result_of_f_nonj = f_nonj(x)
result_of_g_nonj = g_nonj(result_of_f_nonj, 1)
end = time.time()
print(f"Elapsed time: {end - start:.6f} s")

# Check that the results are the same
assert jnp.allclose(result_of_g, result_of_g_nonj)