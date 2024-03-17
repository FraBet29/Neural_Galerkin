import jax
import jax.numpy as jnp


class ProblemData:

    def __init__(self, name, d, domain, T, n, N=2048, dt=0.01):

        # Problem name
        self.name = name

        # Spatial domain
        self.d = d # input dimension
        self.domain = domain # (lower, upper)
        
        # Temporal domain
        self.T = T # final time

        # Discretization
        self.N = N # number of spatial points
        self.dt = dt # time step

        # Sampling
        self.n = n # number of samples
    
    def __str__(self):
        return f'Problem data:\n  name: {self.name}, d: {self.d}, domain: {self.domain}, T: {self.T}, N: {self.N}, dt: {self.dt}, n: {self.n}'
        

class TrainingData:

    def __init__(self, m, batch_size=1000, epochs=1000, gamma=0.1):
            
        # Neural network architecture
        self.m = m # number of nodes
        self.batch_size = batch_size # batch size
        self.epochs = epochs # number of epochs
        self.gamma = gamma # learning rate

    def __str__(self):
        return f'Training data:\n  m: {self.m}, batch size: {self.batch_size}, epochs: {self.epochs}, gamma: {self.gamma}'


####### Define data #######

print('Defining problem data...')
problem_data = ProblemData(name='kdv', d=1, domain=(-20, 40), T=4, n=1000)
print(problem_data)

print('Defining training data...')
training_data = TrainingData(m=10, batch_size=2000, epochs=10000)
print(training_data)

L = 60 # parameter for the gaussian kernel


####### Exact solution for KdV #######

def exactKdVTwoSol(x, t):
    '''
    Function taken from https://github.com/pehersto/ng/solvers/exactKdV.py
    '''

    k = jnp.asarray([1., jnp.sqrt(5.)])
    eta = jnp.asarray([0., 10.73])
    t = jnp.asarray(t)

    etaMat1 = k[0] * x.reshape((-1, 1)) - k[0] ** 3 * t.reshape((1, -1)) + eta[0]
    etaMat2 = k[1] * x.reshape((-1, 1)) - k[1] ** 3 * t.reshape((1, -1)) + eta[1]
    c = ((k[0] - k[1]) / (k[0] + k[1]) )** 2

    f = 1. + jnp.exp(etaMat1) + jnp.exp(etaMat2) + jnp.multiply(jnp.exp(etaMat1), jnp.exp(etaMat2) * c)
    df = k[0] * jnp.exp(etaMat1) + k[1] * jnp.exp(etaMat2) + c * (k[0] + k[1]) * jnp.multiply(jnp.exp(etaMat1), jnp.exp(etaMat2))
    ddf = k[0] ** 2 * jnp.exp(etaMat1) + k[1] ** 2 * jnp.exp(etaMat2) + c * (k[0] + k[1]) ** 2 * jnp.multiply(jnp.exp(etaMat1), jnp.exp(etaMat2))

    y = 2 * jnp.divide(jnp.multiply(f, ddf) - df ** 2, f ** 2)

    y = jnp.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0) # avoid numerical errors far outside of [-1, 2]
    
    return y
