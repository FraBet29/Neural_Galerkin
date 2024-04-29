import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


class Diagnostic:

	def __init__(self):
		self.cond = []
		self.max_eig = []
		self.min_eig = []
		self.eigs_final = None

	def __str__(self):
		return f'Conditioning number: {self.cond[-1]:.2f}, max eigenvalue: {self.max_eig[-1]:.2f}, min eigenvalue: {self.min_eig[-1]:.2f}'
	
	def list2jnp(self):
		cond_jnp, max_eig_jnp, min_eig_jnp = jnp.array(self.cond), jnp.array(self.max_eig), jnp.array(self.min_eig)
		# Check if all eigenvalues with nonzero imaginary part have magnitude close to zero
		assert jnp.all(jnp.abs(min_eig_jnp[min_eig_jnp.imag != 0]) < 1e-6), 'Eigenvalues with nonzero imaginary part detected.'
		# Correct numerical errors
		max_eig_jnp = jnp.real(max_eig_jnp)
		min_eig_jnp = jnp.real(min_eig_jnp)
		if self.eigs_final is not None:
			eigs_final_jnp = jnp.real(self.eigs_final)
			return cond_jnp, max_eig_jnp, min_eig_jnp, eigs_final_jnp
		return cond_jnp, max_eig_jnp, min_eig_jnp, None
	
	def averaged(self):
		cond_jnp, max_eig_jnp, min_eig_jnp, _ = self.list2jnp()
		return jnp.mean(cond_jnp), jnp.mean(max_eig_jnp), jnp.mean(min_eig_jnp)
	
	def plot(self, timesteps):

		cond_jnp, max_eig_jnp, min_eig_jnp, eigs_final_jnp = self.list2jnp()
		
		plt.semilogy(timesteps, cond_jnp)
		plt.title('Conditioning number of M')
		plt.xlabel('t')
		plt.ylabel('cond(M)')
		plt.show()
		
		plt.plot(timesteps, max_eig_jnp)
		plt.plot(timesteps, min_eig_jnp)
		plt.title('Eigenvalues of M')
		plt.xlabel('t')
		plt.ylabel('eigenvalues')
		plt.legend(['max', 'min'])
		plt.show()

		plt.bar(jnp.arange(eigs_final_jnp.shape[0]), jnp.sort(eigs_final_jnp)[::-1])
		plt.title('Eigenvalues of M at the final time')
		plt.xlabel('index')
		plt.ylabel('eigenvalues')
		plt.yscale('log')
		plt.show()


# class Distribution:

# 	def __init__(self, pdf):
# 		self.pdf = pdf
# 		self.dpdf = jax.grad(pdf) # must be evaluated on a scalar

# 	def pdf(self, x):
# 		return self.pdf(x)

# 	def dpdf(self, x):
# 		return self.dpdf(x)


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