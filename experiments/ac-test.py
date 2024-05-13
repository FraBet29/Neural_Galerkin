import numpy as np
import torch
import torch.nn as nn
# from typing import Callable
import matplotlib.pyplot as plt


####### Automatic differentiation #######

def dx(f): # must be evaluated on a tensor!
	return torch.func.grad(f)

def ddx(f): # must be evaluated on a tensor!
	return torch.func.grad(torch.func.grad(f))

##################################################################################################################
############################################## AC equation #######################################################
##################################################################################################################
    
####### Problem data #######

# Space-time domain
X_lower = 0
X_upper = 2 * np.pi
T_lower = 0
T_upper = 6

# PDE parameters
epsilon = 5 * 10 ** (-2)
a = lambda x, t: 1.05 + t * np.sin(x)

# NN architecture
d = 1 # input dimension
m = 2 # nodes
l = 3 # hidden layers
L = 0.5 # parameter for the gaussian kernel
n = 1000 # samples
batch_size = 10 ** 5 # batch size (since n = 1000, we will take the whole dataset?)
epochs = 10 ** 4 # number of epochs
gamma = 0.1 # learning rate

# Initial condition
phi = lambda x, w, b: np.exp(- w ** 2 * np.abs(np.sin(np.pi * (x - b) / L)) ** 2)
u_0 = lambda x: phi(x, np.sqrt(10), 0.5) - phi(x, np.sqrt(10), 4.4)

# Plot the exact initial condition
def plot_exact_initial_condition():
	x_plot = np.linspace(X_lower, X_upper, 100)
	plt.plot(x_plot, u_0(x_plot))
	plt.xlabel('x')
	plt.ylabel('u_0')
	plt.title('Exact initial condition')
	plt.show()

# plot_exact_initial_condition()
	

####### Neural network #######

# Activation function
class PeriodicPhi(nn.Module):

    def __init__(self, d=1, m=1, L=1):
        super(PeriodicPhi, self).__init__()
        self.d = d # input dimension
        self.m = m # number of nodes
        self.L = L
        self.w = torch.nn.Parameter(torch.randn((m))) # random init
        self.b = torch.nn.Parameter(torch.randn((m, d))) # random init

    def forward(self, x):
        # x.shape = (batch, d)
        # b.shape = (m, d)
        x_ext = x.unsqueeze(1) # x_ext.shape = (batch, 1, d)
        phi = torch.exp(- self.w ** 2 * torch.linalg.norm(torch.sin(np.pi * torch.add(x_ext, - self.b) / self.L), dim=2) ** 2)
        return phi # phi.shape = (batch, m)
	
    def string(self):
        return "Phi(d={}, m={}, L={})".format(self.d, self.m, self.L)


# Neural network
deep_net = nn.Sequential(
    PeriodicPhi(d, m, L),
    nn.Tanh(),
    nn.Linear(m, m, L),
    nn.Tanh(),
    nn.Linear(m, m, L),
    nn.Tanh(),
    nn.Linear(m, 1, bias=False)
)


print(deep_net)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(deep_net.parameters(), lr=gamma, weight_decay=1e-6)


####### Initialization #######

# Select device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move the model to the device
deep_net.to(device)

# Dataset definition
class CreateDataset(torch.utils.data.Dataset):

    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __getitem__(self, index):
        sample = {
            'point': self.x[index],
            'solution': self.y[index]}
        return sample

    def __len__(self):
        return len(self.x)


def train_epoch(dataloader):

    deep_net.train()
     
    for batch_idx, data in enumerate(dataloader):
        # Select data for current batch
        x_batch = data['point']
        U_true_batch = data['solution']
        x_batch = x_batch.to(device)
        U_true_batch = U_true_batch.to(device)
        # Forward pass
        U_batch = deep_net(x_batch) # U.shape = (n, d)
        loss = criterion(U_batch, U_true_batch)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        # Update weights
        optimizer.step()

    return loss.item()


def initialize_parameters():
	
	# Find the initial parameters by fitting a least squares problem
	# Equivalently, train the shallow neural network at time t = 0

    # Each batch should contain n = 1000 points (?)
	
	# Sample points uniformly in [0, 1]
    x = np.random.rand(n * 1, d)

    # Define target
    U_true = u_0(x)

    # Define dataset and dataloader
    dataset = CreateDataset(x, U_true)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    losses = []

	# Fit the initial condition
    print("Fitting the initial condition...")
    for epoch in range(epochs):
        loss = train_epoch(dataloader)
        losses.append(loss)
        if epoch % 1000 == 0:
            print("	Epoch: {}/{} - Loss: {}".format(epoch, epochs, loss))

    # Plot the loss
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    # Move model to the CPU
    deep_net.cpu()

    # Plot the true and fitted initial conditions
    x_plot = torch.linspace(X_lower, X_upper, 100).reshape(-1, 1)
    with torch.no_grad():
        U = deep_net(x_plot)
    plt.plot(x_plot.numpy(), u_0(x_plot), label='True')
    plt.plot(x_plot.numpy(), U.numpy(), label='Fitted')
    plt.legend()
    plt.show()

    # Compute relative error
    relative_error = torch.linalg.norm(U - u_0(x_plot)) / torch.linalg.norm(u_0(x_plot))
    print("Relative error:", relative_error)

initialize_parameters()