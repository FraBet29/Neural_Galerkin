import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
from time import time
from NeuralNetwork import *


def loss_fn_wrapper(model, xs, us):

    def loss_fn(params):

        # Define the squared loss for a single pair (x, y)
        def mse(x, u):
            pred = model.apply(params, x)
            loss = jnp.inner(u - pred, u - pred) / 2.0
            return loss

        # Batched version via vmap
        return jnp.mean(jax.vmap(mse)(xs, us), axis=0)

    return jax.jit(loss_fn) # pure function


def init_neural_galerkin(net, problem_data, training_data, theta_init=None):
    '''
    Find the initial parameters by training a neural network at t = 0.
    '''

    # Generate random keys and input data
    key1, key2 = jax.random.split(jax.random.key(training_data.seed))
    x_init = jax.random.uniform(key1, (training_data.batch_size, problem_data.d), 
                                minval=problem_data.domain[0], maxval=problem_data.domain[1])

    # Define the target
    u_true = problem_data.initial_fn(x_init)

    # Initialize the model
    if theta_init == None:
        theta_init = net.init(key2, x_init)

    # Define dataset and dataloader
    # dataset = CreateDataset(x_init, u_true)
    # dataloader = DataLoader(dataset, batch_size=n, collate_fn=collate_fn, shuffle=True)

    # Define the loss function
    mse_loss = loss_fn_wrapper(net, x_init, u_true)
    value_and_grad_fn = jax.value_and_grad(mse_loss)

    # Define the optimizer
    if training_data.scheduler is None:
        opt = optax.adam(training_data.gamma)
        # opt = optax.noisy_sgd(training_data.gamma)
        # opt = jaxopt.LBFGS(fun=mse_loss) # https://jaxopt.github.io/stable/_autosummary/jaxopt.LBFGS.html#jaxopt.LBFGS
    else:
        opt = optax.adam(training_data.scheduler)
        # opt = optax.noisy_sgd(training_data.scheduler)
        # opt = jaxopt.LBFGS(fun=mse_loss)
    opt_state = opt.init(theta_init)
    # opt_state = opt.init_state(theta_init)

    # Define a TrainState
    # state = train_state.TrainState.create(apply_fn=net.apply, tx=opt, params=theta_init['params'])

    losses = []

    # Fit the initial condition
    print('Fitting the initial condition...')
    timer = time()

    for epoch in range(training_data.epochs):
        loss, grads = value_and_grad_fn(theta_init)
        updates, opt_state = opt.update(grads, opt_state)
        # updates, opt_state = opt.update(grads, opt_state, theta_init)
        theta_init = optax.apply_updates(theta_init, updates)
        # theta_init, opt_state = opt.update(grads, opt_state)
        losses.append(loss)

        if epoch % 1000 == 0:
            err = jnp.linalg.norm(u_true - net.apply(theta_init, x_init)) / jnp.linalg.norm(u_true)
            print(f'epoch {epoch}, loss = {loss}, error = {err}')
    
    print(f'Elapsed time: {time() - timer:.0f}s')

    # Plot the loss
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Initial fit - Loss')
    plt.show()

    # Plot the true and fitted initial conditions
    x_plot = jnp.linspace(problem_data.domain[0], problem_data.domain[1], problem_data.N)
    u_pred = net.apply(theta_init, x_plot.reshape(-1, 1))
    plt.plot(x_plot, problem_data.initial_fn(x_plot), label='True')
    plt.plot(x_plot, u_pred, label='Fitted')
    plt.title('Initial fit - True vs Fitted')
    plt.legend()
    plt.show()

    # Compute the relative error
    relative_error = jnp.linalg.norm(u_pred.squeeze() - problem_data.initial_fn(x_plot)) / jnp.linalg.norm(problem_data.initial_fn(x_plot))
    print("Relative error of the initial fit:", relative_error)

    return theta_init