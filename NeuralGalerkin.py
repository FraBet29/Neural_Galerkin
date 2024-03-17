from Data import *
from NeuralNetwork import *
from InitialFit import *
from AssembleSystem import *
import matplotlib.pyplot as plt
from matplotlib import colormaps
import scipy


####### Initialization #######

def plot_exact_initial_condition():
	x_plot = jnp.linspace(problem_data.domain[0], problem_data.domain[1], problem_data.N)
	plt.plot(x_plot, exactKdVTwoSol(x_plot, 0))
	plt.xlabel('x')
	plt.ylabel('u_0')
	plt.title('Exact initial condition')
	plt.show()

# plot_exact_initial_condition()

if run_training:
	
	# Define the neural network
	shallow_net = ShallowNet(problem_data.d, training_data.m, L)
	
    # Fit the initial condition
	theta_init = init_neural_galerkin(shallow_net)

	# Save the initial parameters
	jnp.save('./data/theta_init_' + problem_data.name + '.npy', theta_init)


