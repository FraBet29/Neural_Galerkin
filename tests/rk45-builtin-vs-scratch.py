import numpy as np
from scipy.integrate import RK45
import time

# Define the system of ODEs
def system(t, x):
    n = len(x)  # Dimension of the system
    M = 0.5 * np.eye(n)  # Example matrix M
    # M = np.array([[1, 0.5], [0.5, 1]])  # Example matrix M
    # F = np.array([t, np.sin(t)])         # Example vector F
    F = t * np.linspace(0, 1, n)  # Example vector F
    dxdt = np.linalg.solve(M, F)         # Solve for dx/dt
    return dxdt

n = 10  # Dimension of the system

# Initial conditions
t0 = 0
x0 = np.zeros(n)  # Initial values of x1 and x2

# Integration bounds
t_bound = 10

### METHOD 1: USING SCIPY'S RK45 INTEGRATOR ###

# Create RK45 integrator object
integrator = RK45(system, t0, x0, t_bound, max_step=0.001)

# Lists to store results
t_values = [t0]
x_values = [x0]

# Integrate
start = time.time()
while integrator.t < t_bound:
    integrator.step()
    t_values.append(integrator.t)
    x_values.append(integrator.y)
end = time.time()
print(f"Time taken: {end - start:.2f} seconds")

# Convert results to numpy arrays
t_values = np.array(t_values)
x_values = np.array(x_values)

# Print the final values of x1 and x2
print("Final values:")
for i in range(n):
	print(f"x{i+1}({t_values[-1]:.2f}) = {x_values[-1, i]:.4f}")

### METHOD 2: IMPLEMENTING RK45 FROM SCRATCH ###

def rk45_step(system, t, x, dt):
	k1 = system(t, x)
	k2 = system(t + 0.5*dt, x + 0.5*dt*k1)
	k3 = system(t + 0.5*dt, x + 0.5*dt*k2)
	k4 = system(t + dt, x + dt*k3)
	x_next = x + dt*(k1 + 2*k2 + 2*k3 + k4)/6
	return x_next

# Re-initialize initial conditions
t = t0
x = x0
dt = 0.001  # Time step

# Lists to store results
t_values_scratch = [t0]
x_values_scratch = [x0]

# Integrate
start = time.time()
while t < t_bound:
	x = rk45_step(system, t, x, dt)
	t += dt
	t_values_scratch.append(t)
	x_values_scratch.append(x)
end = time.time()
print(f"\nTime taken (from scratch): {end - start:.2f} seconds")
      
# Convert results to numpy arrays
t_values_scratch = np.array(t_values_scratch)
x_values_scratch = np.array(x_values_scratch)

# Print the final values of x1 and x2
print("Final values (from scratch):")
for i in range(n):
	print(f"x{i+1}({t_values_scratch[-1]:.2f}) = {x_values_scratch[-1, i]:.4f}")
