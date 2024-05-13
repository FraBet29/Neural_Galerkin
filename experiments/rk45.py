import numpy as np
from scipy.integrate import RK45
import matplotlib.pyplot as plt

# Define the ODE function
def ode_function(t, y):
    return -0.5 * y

# Define initial conditions
t0 = 0
y0 = np.array([1])

# Define integration bounds
t_bound = 10

# Create RK45 integrator object
integrator = RK45(ode_function, t0, y0, t_bound)

# Lists to store results
t_values = [t0]
y_values = [y0]

# Integrate until the desired t_bound is reached
while integrator.t < t_bound:
    integrator.step()
    t_values.append(integrator.t)
    y_values.append(integrator.y)

# Print time steps
print(t_values)

# Plot the solution
plt.plot(t_values, y_values, label='Numerical solution (RK45)')
plt.xlabel('t')
plt.ylabel('y')
plt.title('Solution of dy/dt = -0.5y')
plt.legend()
plt.grid(True)
plt.show()
