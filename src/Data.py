class ProblemData:

    def __init__(self, name, d, domain, T, initial_fn, exact_sol, N=2048, dt=0.01):
        self.name = name # problem name
        self.d = d # input dimension
        self.domain = domain # 1D: (lower, upper)
        self.T = T # final time
        self.initial_fn = initial_fn # initial condition
        self.exact_sol = exact_sol # exact solution (if available)
        self.N = N # number of spatial points
        self.dt = dt # time step (if needed)
    
    def __str__(self):
        return f'Problem data:\n  name: {self.name}, d: {self.d}, domain: {self.domain}, T: {self.T}, N: {self.N}, dt: {self.dt}'
        

class TrainingData:

    def __init__(self, m, batch_size=1000, epochs=1000, gamma=0.1, seed=0, scheduler=None):
        self.m = m # number of neurons per layer
        self.batch_size = batch_size # batch size
        self.epochs = epochs # number of epochs
        self.gamma = gamma # learning rate
        self.seed = seed # seed for reproducibility
        self.scheduler = scheduler # learning rate scheduler

    def __str__(self):
        return f'Training data:\n  m: {self.m}, batch size: {self.batch_size}, epochs: {self.epochs}, gamma: {self.gamma}, seed: {self.seed}, scheduler: {self.scheduler}'
