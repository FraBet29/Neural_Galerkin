import jax
import jax.numpy as jnp
import time


def generate_matrix(m, n):
	matrix = jax.random.uniform(jax.random.PRNGKey(0), shape=(m, n))
	return matrix


def gram_schmidt_built_in(matrix):
	# orthonormal_basis = jnp.linalg.qr(matrix)[0]
	orthonormal_basis = jax.scipy.linalg.qr(matrix, mode='economic')[0]
	return orthonormal_basis


def orthogonalize(U, eps=1e-15):
    """
	# https://gist.github.com/anmolkabra/b95b8e7fb7a6ff12ba5d120b6d9d1937
    Orthogonalizes the matrix U (m x n) using Gram-Schmidt.
    If the columns of U are linearly dependent with rank(U) = r, the last n-r columns will be 0.
    """
    # jax.config.update('jax_enable_x64', True) # enable double precision
    n = U.shape[1]
    V = U.T
    for i in range(n):
        V_prev = V[0:i]
        proj = jnp.dot(V_prev, V[i].T)
        V = V.at[i].add(-jnp.dot(proj, V_prev).T)
        # V = V.at[i].add(-jnp.dot(jnp.dot(V[0:i], V[i].T), V[0:i]).T)
        if jnp.linalg.norm(V[i]) < eps:
            # V = V.at[i][V[i] < eps].set(0.0)
            V = V.at[i].set(0.0)
        else:
            V = V.at[i].divide(jnp.linalg.norm(V[i]))
    # Compute rank
    # rank = jnp.sum(jnp.linalg.norm(V, axis=1) > eps)
    return V.T


# https://github.com/tensorflow/probability/blob/v0.23.0/tensorflow_probability/python/math/gram_schmidt.py#L28

def gram_schmidt(vectors):
	"""Implementation of the modified Gram-Schmidt orthonormalization algorithm.

	We assume here that the vectors are linearly independent. Zero vectors will be
	left unchanged, but will also consume an iteration against `num_vectors`.

	Args:
	vectors: A Tensor of shape `[d, n]` of `d`-dim column vectors to orthonormalize.

	Returns:
	A Tensor of shape `[d, n]` corresponding to the orthonormalization.
	"""
	n = vectors.shape[-1]

	def cond_fn(state):
		vecs, i = state
		return i < n - 1

	def body_fn(state):
		vecs, i = state
		# Slice out the vector w.r.t. which we're orthogonalizing the rest.
		vec_norm = jnp.linalg.norm(vecs[:, i])
		u = jnp.divide(vecs[:, i], vec_norm) # (d, )
		# Find weights by dotting the d x 1 against the d x n.
		weights = jnp.einsum('dm,dn->n', u[:, jnp.newaxis], vecs) # (n, )
		# Project out vector `u` from the trailing vectors.
		masked_weights = jnp.where(jnp.arange(n) > i, weights, 0.)[jnp.newaxis, :] # (1, n)
		vecs = vecs - jnp.multiply(u[:, jnp.newaxis], masked_weights) # (d, n)
		vecs = jnp.where(jnp.isnan(vecs), 0.0, vecs)
		vecs = jnp.reshape(vecs, vectors.shape)
		return vecs, i + 1

	vectors, _ = jax.lax.while_loop(cond_fn, body_fn, (vectors, jnp.zeros([], dtype=jnp.int32))) # call body_fn while cond_fn is true
	vec_norm = jnp.linalg.norm(vectors, ord=2, axis=0, keepdims=True)
	vectors = jnp.divide(vectors, vec_norm)
	return jnp.where(jnp.isnan(vectors), 0.0, vectors)


@jax.jit
def gram_schmidt_poly(u_dth, P_proj):
	"""Implementation of the modified Gram-Schmidt orthonormalization algorithm.
	Adapted from: https://github.com/tensorflow/probability/blob/v0.23.0/tensorflow_probability/python/math/gram_schmidt.py#L28
	"""
	n = u_dth.shape[-1]

	def body_fn(i, vecs):
		# Slice out the vector w.r.t. which we're orthogonalizing the rest.
		vec_norm = jnp.linalg.norm(vecs[:, i])
		u = jnp.divide(vecs[:, i], vec_norm) # (d, )
		# Find weights by dotting the d x 1 against the d x n.
		# weights = jnp.einsum('dm,dn->n', u[:, jnp.newaxis], vecs) # (n, )
		# weights = P_proj[i] # (n, )
		weights = jnp.divide(P_proj[i], jnp.einsum('ii->i', P_proj)) # (n, )
		# Project out vector `u` from the trailing vectors.
		masked_weights = jnp.where(jnp.arange(n) > i, weights, 0.)[jnp.newaxis, :] # (1, n)
		vecs = vecs - jnp.multiply(u[:, jnp.newaxis], masked_weights) # (d, n)
		vecs = jnp.where(jnp.isnan(vecs), 0.0, vecs)
		vecs = jnp.reshape(vecs, u_dth.shape)
		return vecs
	
	u_dth = jax.lax.fori_loop(0, n, body_fn, u_dth)
	vec_norm = jnp.linalg.norm(u_dth, ord=2, axis=0, keepdims=True)
	u_dth = jnp.divide(u_dth, vec_norm)
	return jnp.where(jnp.isnan(u_dth), 0.0, u_dth)


def check_orthonormality(matrix):
	for i in range(matrix.shape[1]):
		for j in range(matrix.shape[1]):
			inner_product = jnp.dot(matrix[:, i], matrix[:, j])
			# print(inner_product)
			if (i == j and not jnp.isclose(inner_product, 1.0, atol=1e-6)) or (i != j and not jnp.isclose(inner_product, 0.0, atol=1e-6)):
				print("Matrix is not orthonormal")
				return
	print("Matrix is orthonormal")
	# orthonormality = jnp.dot(matrix, matrix.T)
	# # print(orthonormality)
	# if jnp.allclose(orthonormality, jnp.eye(matrix.shape[0]), atol=1e-3):
	# 	print("Matrix is orthonormal")
	# else:
	# 	print("Matrix is not orthonormal")


def main():
	
	# jax.config.update('jax_enable_x64', True) # enable double precision
	jnp.set_printoptions(linewidth=jnp.inf)

	# d = 100 # 100
	# n = 3 # 30
	
	# matrix = generate_matrix(d, n)
	
	# start = time.time()
	# orthonormal_basis = gram_schmidt_built_in(matrix)
	# print("Built-in Gram-Schmidt: ", time.time() - start)
	# # print(orthonormal_basis)
	# check_orthonormality(orthonormal_basis)
	# print(orthonormal_basis.T @ orthonormal_basis)
	
	# start = time.time()
	# orthonormal_basis = orthogonalize(matrix)
	# print("Custom Gram-Schmidt: ", time.time() - start)
	# # print(orthonormal_basis)
	# check_orthonormality(orthonormal_basis)
	# print(orthonormal_basis.T @ orthonormal_basis)

	# start = time.time()
	# orthonormal_basis = gram_schmidt(matrix)
	# print("Modified Gram-Schmidt: ", time.time() - start)
	# # print(orthonormal_basis)
	# check_orthonormality(orthonormal_basis)
	# print(orthonormal_basis.T @ orthonormal_basis)

	# Gram Schmidt for polynomials in [-1, 1]
	p0 = lambda x: 1 + 0. * x
	p1 = lambda x: x
	p2 = lambda x: x ** 2
	p3 = lambda x: x ** 3
	p4 = lambda x: x ** 4
	# Build projection coefficients
	x = jnp.linspace(-1, 1, 3)
	P = jnp.stack([p0(x), p1(x), p2(x), p3(x), p4(x)], axis=1) # (n, 5)
	P_proj = jnp.mean(P[:, :, jnp.newaxis] * P[:, jnp.newaxis, :], axis=0) # (5, 5)
	# Orthonormalize
	start = time.time()
	# orthonormal_basis = gram_schmidt_poly(P, P_proj)
	orthonormal_basis = gram_schmidt(P)
	print("Modified Gram-Schmidt for polynomials: ", time.time() - start)
	print(orthonormal_basis)
	# Try with QR
	# start = time.time()
	# orthonormal_basis = jnp.linalg.qr(P)[0]
	# print("QR for polynomials: ", time.time() - start)
	# print(orthonormal_basis)
	# check_orthonormality(orthonormal_basis)
	# print(orthonormal_basis.T @ orthonormal_basis)


main()