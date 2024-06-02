import jax
import jax.numpy as jnp


def quantile_function(qs, cws, xs):
    """
    Computes the quantile function of an empirical distribution.

    Parameters:
		qs: array-like, shape (n,)
			quantiles at which the quantile function is evaluated
		cws: array-like, shape (m, ...)
			cumulative weights of the 1D empirical distribution, if batched, must be similar to xs
		xs: array-like, shape (n, ...)
			locations of the 1D empirical distribution, batched against the `xs.ndim - 1` first dimensions

    Returns:
		q: array-like, shape (..., n)
			The quantiles of the distribution
    
    """
    n = xs.shape[0]
	# cws = cws.T
	# qs = qs.T
    idx = jnp.searchsorted(cws, qs) # .T
    return jnp.take_along_axis(xs, jnp.clip(idx, 0, n - 1), axis=0)


def wasserstein_1d(u_values, v_values, p=1, require_sort=True):
    """
    Function adapted from: https://github.com/PythonOT/POT/blob/ab12dd6606122dc7804f69b18eaec19adfca9c71/ot/lp/solver_1d.py#L50
    
    Computes the 1 dimensional OT loss [15] between two (batched) empirical distributions.

    Math:
        OT_{loss} = \int_0^1 |cdf_u^{-1}(q) - cdf_v^{-1}(q)|^p dq

    It is formally the p-Wasserstein distance raised to the power p.
    In the 1D case, it can be computed as the sum of the differences between the quantile functions of the two distributions.
    We do so in a vectorized way by first building the individual quantile functions then integrating them.

    Parameters:
		u_values: array-like, shape (n, ...)
			locations of the first empirical distribution
		v_values: array-like, shape (m, ...)
			locations of the second empirical distribution
		p: int, optional
			order of the ground metric used, should be at least 1 (see [2, Chap. 2], default is 1
		require_sort: bool, optional
			sort the distributions atoms locations, if False we will consider they have been sorted prior to being passed to
			the function, default is True

    Returns:
		cost: float/array-like, shape (...)
			the batched EMD

    References:
    	[15] Peyré, G., & Cuturi, M. (2018). Computational Optimal Transport.

    """
    n = u_values.shape[0]
    m = v_values.shape[0]

    u_weights = jnp.full(u_values.shape, 1 / n)
    v_weights = jnp.full(v_values.shape, 1 / m)

    if require_sort:
        
        u_sorter = jnp.argsort(u_values, 0)
        u_values = jnp.take_along_axis(u_values, u_sorter, 0)

        v_sorter = jnp.argsort(v_values, 0)
        v_values = jnp.take_along_axis(v_values, v_sorter, 0)

        u_weights = jnp.take_along_axis(u_weights, u_sorter, 0)
        v_weights = jnp.take_along_axis(v_weights, v_sorter, 0)

    u_cumweights = jnp.cumsum(u_weights, 0)
    v_cumweights = jnp.cumsum(v_weights, 0)

    qs = jnp.sort(jnp.concatenate((u_cumweights, v_cumweights), 0), 0)
    
    u_quantiles = quantile_function(qs, u_cumweights, u_values)
    v_quantiles = quantile_function(qs, v_cumweights, v_values)
    
    qs = jnp.pad(qs, pad_width=[(1, 0)] + (qs.ndim - 1) * [(0, 0)], mode='constant', constant_values=0.0)
    delta = qs[1:, ...] - qs[:-1, ...]
    diff_quantiles = jnp.abs(u_quantiles - v_quantiles)

    if p == 1:
        return jnp.sum(delta * diff_quantiles, axis=0)
    
    return jnp.sum(delta * jnp.power(diff_quantiles, p), axis=0)
