import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


class Diagnostic:

	def __init__(self):
		self.cond = []
		self.max_eig = []
		self.min_eig = []

	def __str__(self):
		return f'Conditioning number: {self.cond[-1]:.2f}, max eigenvalue: {self.max_eig[-1]:.2f}, min eigenvalue: {self.min_eig[-1]:.2f}'
	
	def list2jnp(self):
		cond_jnp, max_eig_jnp, min_eig_jnp = jnp.array(self.cond), jnp.array(self.max_eig), jnp.array(self.min_eig)
		# Check if all eigenvalues with nonzero imaginary part have magnitude close to zero
		assert jnp.all(jnp.abs(min_eig_jnp[min_eig_jnp.imag != 0]) < 1e-6), 'Eigenvalues with nonzero imaginary part detected.'
		# Correct for numerical errors
		max_eig_jnp = jnp.real(max_eig_jnp)
		min_eig_jnp = jnp.real(min_eig_jnp)
		return cond_jnp, max_eig_jnp, min_eig_jnp
	
	def averaged(self):
		cond_jnp, max_eig_jnp, min_eig_jnp = self.list2jnp()
		return jnp.mean(cond_jnp), jnp.mean(max_eig_jnp), jnp.mean(min_eig_jnp)
	
	def plot(self, timesteps):

		cond_jnp, max_eig_jnp, min_eig_jnp = self.list2jnp()
		
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