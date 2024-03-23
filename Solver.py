import jax
import jax.numpy as jnp
import optax
import scipy


def backward_euler_scheme(theta_flat, x, t, r_loss):
	'''
	Implicit Euler scheme with ADAM. 
	'''
	theta_flat_k = jnp.copy(theta_flat)
	optimizer = optax.adam(1e-3)
	opt_state = optimizer.init(theta_flat)
	
	for _ in range(50):
		# if i % 10 == 0:
		#     print(f'Adam iter: {i}/100')
		#     print(r_loss(theta_flat, theta_flat_k, x, t))
		grads = jax.grad(r_loss)(theta_flat, theta_flat_k, x, t)
		updates, opt_state = optimizer.update(grads, opt_state)
		theta_flat = optax.apply_updates(theta_flat, updates)

	# print(f'Implicit Euler - ADAM residual: {r_loss(theta_flat, theta_flat_k, x, t):.5f}')
    
	return theta_flat


def runge_kutta_scheme(theta_flat, x, t, dt, M_fn, F_fn):
    
    def rhs_RK45(theta_flat, x, t, M_fn, F_fn):
        return jnp.linalg.lstsq(M_fn(theta_flat, x), F_fn(theta_flat, x, t))[0]
    
    def rhs_RK45_wrapper(t, theta_flat):
        return rhs_RK45(theta_flat, x, t, M_fn, F_fn)
    
    theta_flat_k = jnp.copy(theta_flat)
    scheme = scipy.integrate.RK45(rhs_RK45_wrapper, t, theta_flat_k, t + dt)
    scheme.step()
    
    return scheme.y