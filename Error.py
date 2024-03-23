import jax
import jax.numpy as jnp


def compute_error(solution, exact_solution, problem_data):

    x_plot = jnp.linspace(problem_data.domain[0], problem_data.domain[1], problem_data.N)
    t_plot = jnp.linspace(0, problem_data.T, int(problem_data.T / problem_data.dt) + 1)
    ref_solution = exact_solution(x_plot, t_plot).T
    space_time_solution = jnp.array(solution) # (time, space)

    diff = ref_solution - space_time_solution
    cumulative_ref_norms = jnp.cumsum(jnp.linalg.norm(ref_solution, axis=1))
    cumulative_diff_norms = jnp.cumsum(jnp.linalg.norm(diff, axis=1))
    errors = cumulative_diff_norms / cumulative_ref_norms
    errors = jnp.nan_to_num(errors, nan=0.0) # handle division by zero
    errors = errors.tolist()

    return errors
