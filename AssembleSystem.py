import jax
import jax.numpy as jnp
from Data import *
from JaxUtils import *


#@jax.jit
def rhs_fn(x, t, theta_flat, U, U_dx, U_dddx):
	u = U(theta_flat, x)
	u_x = U_dx(theta_flat, x)
	u_xxx = U_dddx(theta_flat, x)
	return - u_xxx - 6 * u * u_x


#@jax.jit
def assemble_fn(x, t, theta_flat, U_dtheta):
	u_dth = U_dtheta(theta_flat, x)
	M = jnp.mean(u_dth[:, :, jnp.newaxis] * u_dth[:, jnp.newaxis, :], axis=0)
	f = rhs_fn(x, t) # source term
	F = jnp.mean(u_dth * f[:, jnp.newaxis], axis=0)
	return M, F


#@jax.jit
def M_fn(x, theta_flat, U_dtheta):
	u_dth = U_dtheta(theta_flat, x)
	M = jnp.mean(u_dth[:, :, jnp.newaxis] * u_dth[:, jnp.newaxis, :], axis=0)
	return M


#@jax.jit
def F_fn(x, t, theta_flat, U, U_dx, U_dddx, U_dtheta):
	u_dth = U_dtheta(theta_flat, x)
	f = rhs_fn(x, t, theta_flat, U, U_dx, U_dddx)
	F = jnp.mean(u_dth[:, :] * f[:, jnp.newaxis], axis=0)
	return F


class AssembleSystem():
	'''
	Class to assemble the system of equations.
	'''
	
	def __init__(self, net, theta):
		
		theta_flat, unravel = jax.flatten_util.ravel_pytree(theta) # ravel (flatten) a pytree of arrays down to a 1D array
		u_scalar = unraveler(net.apply, unravel)

		self.theta = theta
		self.theta_flat = theta_flat
		self.u_scalar = u_scalar
		self.unravel = unravel

		self.U, self.U_dtheta, self.U_dx, self.U_dddx = self.batch()


	def batch(self):
		
		# Solution
		U = jax.vmap(self.u_scalar, (None, 0)) # jax.vmap(fun, in_axes)

		# Derivative w.r.t. theta
		U_dtheta = jax.vmap(jax.grad(self.u_scalar), (None, 0))

		# Spatial derivatives
		U_dx = jax.vmap(gradsqz(self.u_scalar, 1), (None, 0))
		U_dddx = jax.vmap(gradsqz(gradsqz(gradsqz(self.u_scalar, 1), 1), 1), (None, 0))
		
		return U, U_dtheta, U_dx, U_dddx
	

	def update(self, theta_flat):
		self.theta = self.unravel(theta_flat)
		self.theta_flat = theta_flat
		self.u_scalar = unraveler(self.u_scalar, self.unravel)
		self.U, self.U_dtheta, self.U_dx, self.U_dddx = self.batch()

	
	def rhs(self, x, t):
		'''
		Source term for the KdV equation.
		Args:
			x: jnp.array, input data
			t: float, time
		Returns:
			array, source term
		'''
		# u = self.U(self.theta_flat, x)
		# u_x = self.U_dx(self.theta_flat, x)
		# u_xxx = self.U_dddx(self.theta_flat, x)
		# return - u_xxx - 6 * u * u_x
		return rhs_fn(x, t, self.theta_flat, self.U, self.U_dx, self.U_dddx)
	

	def assemble(self, x, t):
		# u_dth = self.U_dtheta(self.theta_flat, x)
		# M = jnp.mean(u_dth[:, :, jnp.newaxis] * u_dth[:, jnp.newaxis, :], axis=0)
		# f = self.rhs(x, t) # source term
		# F = jnp.mean(u_dth * f[:, jnp.newaxis], axis=0)
		# return M, F
		return assemble_fn(x, t, self.theta_flat, self.U_dtheta)


	def assemble_M(self, x):
		# u_dth = self.U_dtheta(self.theta_flat, x)
		# M = jnp.mean(u_dth[:, :, jnp.newaxis] * u_dth[:, jnp.newaxis, :], axis=0)
		# return M
		return M_fn(x, self.theta_flat, self.U_dtheta)
	

	def assemble_F(self, x, t):
		# u_dth = self.U_dtheta(self.theta_flat, x)
		# f = self.rhs(x, t)
		# F = jnp.mean(u_dth[:, :] * f[:, jnp.newaxis], axis=0)
		# return F
		return F_fn(x, t, self.theta_flat, self.U, self.U_dx, self.U_dddx, self.U_dtheta)


	def assemble_r(self, theta_flat_k, x, t):
		return jnp.dot(self.assemble_M(x), self.theta) - jnp.dot(self.assemble_M(x), theta_flat_k) - \
			problem_data.dt * self.assemble_F(x, t + problem_data.dt)


	def r_loss(self, theta_flat_k, x, t):
		return jnp.linalg.norm(self.assemble_r(theta_flat_k, x, t))
	