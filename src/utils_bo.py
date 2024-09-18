import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from gpax.models.gp import ExactGP
from gpax.acquisition import EI


# Define the objective function
def objective_function(x):
    from test_functions import sinusoidal_synthetic  # Replace with your actual function
    return sinusoidal_synthetic(x)


# Define the experiment settings as a dictionary
experiment_settings = {
    "search_space": np.array([[-10], [10]]),  # 1D search space example
    "num_iterations": 50,                     # Number of optimization iterations
    "num_initial_points": 5,                  # Number of initial observation points
    "num_samples": 10,                        # Number of samples for acquisition function
    "kernel": "RBF",                          # Kernel for GP model
    "objective_function": objective_function  # Actual objective function
}

# Extract settings from the dictionary
search_space = experiment_settings["search_space"]
num_iterations = experiment_settings["num_iterations"]
num_initial_points = experiment_settings["num_initial_points"]
num_samples = experiment_settings["num_samples"]
kernel = experiment_settings["kernel"]
objective_function = experiment_settings["objective_function"]

# Step 1: Generate random initial observation points within the search space
X_init = np.random.uniform(search_space[0], search_space[1], (num_initial_points, search_space.shape[1]))
y_init = np.array([objective_function(x) for x in X_init])

# Print initial data
print("Initial X:", X_init)
print("Initial y:", y_init)

# Step 2: Initialize the GP model
rng_key = random.PRNGKey(0)  # JAX random number generator
gp_model = ExactGP(input_dim=search_space.shape[1], kernel=kernel)  # Using specified kernel
gp_model.fit(rng_key, jnp.array(X_init), jnp.array(y_init))

# Step 3: Main loop for Bayesian optimization
for i in range(num_iterations):
    # Step 3.1: Generate candidate points within the search space
    X_candidates = jnp.linspace(search_space[0], search_space[1], 100).reshape(-1, search_space.shape[1])

    # Step 3.2: Compute acquisition values for all candidate points
    acq_values = jnp.array([EI(rng_key, gp_model, X, best_f=jnp.min(jnp.array(y_init)),
                               maximize=False, n=num_samples) for X in X_candidates])

    # Step 3.3: Select the candidate point with the highest acquisition value
    X_next = X_candidates[jnp.argmax(acq_values)]

    # Step 3.4: Evaluate the objective function at the selected point
    y_next = objective_function(X_next)

    # Step 3.5: Reshape and add the new data point to the dataset
    X_init = np.vstack((X_init, np.array(X_next)))
    y_init = np.vstack((y_init, np.array(y_next).reshape(-1, 1)))

    # Step 3.6: Re-train the GP model with the updated dataset
    gp_model.fit(rng_key, jnp.array(X_init), jnp.array(y_init))

    # Step 3.7: Display progress
    print(f"Iteration {i + 1}: X_next = {X_next}, y_next = {y_next}, EI = {acq_values[jnp.argmax(acq_values)]}")

# The final result is in X_init and y_init, containing all evaluated points and their function values.
