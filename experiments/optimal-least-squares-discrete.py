import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import scipy


# ENABLE PRINT ON MORE LINES
jnp.set_printoptions(threshold=jnp.inf, linewidth=jnp.inf)

# ENABLE DOUBLE PRECISION
# jax.config.update('jax_enable_x64', True)


# define a least squares problem min ||Ax - b||^2 where A and b are NOT functions of x, but just vectors

n = 20
p = 5
key = jax.random.PRNGKey(0)

A = 1./(jnp.arange(1, p+1) + jnp.arange(1, n+1)[:, jnp.newaxis])
b = jax.random.normal(key, (n, ))

# x = jnp.linspace(0, 2 * jnp.pi, n)
x = 2 * jnp.pi * jax.random.uniform(key, (n, ))

# suppose first that the columns of A are linearly independent

print('A.shape =', A.shape)
print('b.shape =', b.shape)
print('rank(A) =', jnp.linalg.matrix_rank(A))

# try to solve the least squares problem directly

# v = jnp.linalg.lstsq(A(x), b(x))[0]
# print('error (original) =', jnp.linalg.norm(jnp.dot(A(x), v) - b(x)))

# the associated normal equations are ill conditioned so it is not a good idea to solve them directly

M = jnp.mean(A[:, :, jnp.newaxis] * A[:, jnp.newaxis, :], axis=0) # (p, p)
F = jnp.mean(A[:, :] * b[:, jnp.newaxis], axis=0) # (p, )

print('M.shape =', M.shape)
print('cond(M) =', jnp.linalg.cond(M))

v = jnp.linalg.solve(M, F)

print('error (normal equations) =', jnp.linalg.norm(jnp.dot(M, v) - F))

# sanity check for the normal equations

v_check = jnp.linalg.lstsq(A, b)[0]

print('error (normal equations check pt. 1) =', jnp.linalg.norm(jnp.dot(A, v) - b))
print('error (normal equations check pt. 2) =', jnp.linalg.norm(jnp.dot(A, v_check) - b))

# OK UNTIL NOW!

# we want to orthogonalize the columns of A to form an orthonormal basis for the column space of A
# to do so we use gram-schmidt with the Euclidean inner product

# first try with built-in QR

Q_bi, _ = jnp.linalg.qr(A) # (n, p)

# then try with modified gram-schmidt

@jax.jit
def orthogonalize(u_dth):
	'''
	Implementation of the modified Gram-Schmidt orthonormalization algorithm.
	Adapted from: https://github.com/tensorflow/probability/blob/v0.23.0/tensorflow_probability/python/math/gram_schmidt.py#L28
	'''
	n = u_dth.shape[-1] # actually, p (just a problem with names)

	def body_fn(i, vecs): # the initial vectors in vecs are progressively replaced by the orthonormalized ones
		vec_norm = jnp.linalg.norm(vecs[:, i])
		u = jnp.divide(vecs[:, i], vec_norm) # (d, )
		# u = vecs[:, i] # (d, )
		weights = jnp.einsum('dm,dn->n', u[:, jnp.newaxis], vecs) # (n, )
        # weights = M_proj[i] # (n, )
		# weights = jnp.divide(M_proj[i], jnp.einsum('ii->i', M_proj)) # (n, ) # <u_dth_i, u_dth_j> / <u_dth_i, u_dth_i>
		masked_weights = jnp.where(jnp.arange(n) > i, weights, 0.)[jnp.newaxis, :] # (1, n) # consider only the first i vectors
		vecs = vecs - jnp.multiply(u[:, jnp.newaxis], masked_weights) # (d, n)
		vecs = jnp.where(jnp.isnan(vecs), 0.0, vecs)
		vecs = jnp.reshape(vecs, u_dth.shape)
		return vecs

	u_dth = jax.lax.fori_loop(0, n, body_fn, u_dth)
	vec_norm = jnp.linalg.norm(u_dth, ord=2, axis=0, keepdims=True)
	u_dth = jnp.divide(u_dth, vec_norm)
	return jnp.where(jnp.isnan(u_dth), 0.0, u_dth), vec_norm


Q, Q_norms = orthogonalize(A)

print('Q.shape =', Q.shape)

# check that Q and Q_bi provide the same orthonormal basis

print('relative error QR vs GS: ', jnp.linalg.norm(Q - Q_bi) / jnp.linalg.norm(Q_bi))

Q = Q_bi

# once we have an orthonormal basis we can define weighted estimators

w = p / jnp.sum(Q ** 2, axis=1) # (n, )

G = jnp.mean(w[: , jnp.newaxis, jnp.newaxis] * (Q[:, :, jnp.newaxis] * Q[:, jnp.newaxis, :]), axis=0) # (p, p)
d = jnp.mean(w[:, jnp.newaxis] * (Q[:, :] * b[:, jnp.newaxis]), axis=0) # (p, )

print('G.shape =', G.shape)
print('d.shape =', d.shape)

print('cond(G) =', jnp.linalg.cond(G))

v_weighted = jnp.linalg.solve(G, d)

print('error (weighted) =', jnp.linalg.norm(jnp.dot(G, v_weighted) - d))

# OK UNTIL NOW (ASSUMING GS WORKS CORRECTLY)!

# recover the gram-schmidt change of basis matrix

# B_GS_diag = jnp.sqrt(jnp.mean(A(x) ** 2, axis=0))
B_GS_diag = Q_norms
B_GS_off = jnp.mean(A[:, :, jnp.newaxis] * Q[:, jnp.newaxis, :], axis=0)
# B_GS = jnp.diag(B_GS_diag) + jnp.hstack((jnp.zeros((p, 1)), jnp.triu(B_GS_off, k=0)[:, :-1]))
# B_GS = jnp.linalg.inv(B_GS) # CHANGE THIS!
B_GS = jnp.diag(B_GS_diag) + jnp.hstack((jnp.zeros((p, 1)), jnp.triu(B_GS_off, k=0)[:, :-1])).T

print('cond(B_GS) =', jnp.linalg.cond(B_GS))

# recover the original solution

# v_recovered = jnp.dot(B_GS.T, v_weighted)
v_recovered = jnp.linalg.solve(B_GS, v_weighted)
print('error (normal equations check pt. 3) =', jnp.linalg.norm(jnp.dot(A, v_recovered) - b))

# print('v =', v)
# print('v_recovered =', v_recovered)