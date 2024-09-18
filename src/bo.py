import jax
import jax.numpy as jnp
import numpy as np
from gpax.acquisition import EI, optimize_acq
from gpax.models.gp import ExactGP
from jax import random

from utils_bo import DataTransformer, generate_initial_data

# Define the objective function
def objective_function(x):
    from test_functions import \
        sinusoidal_synthetic  # Replace with your actual function

    return sinusoidal_synthetic(x)


# Define the main function for running Bayesian Optimization
def run_bo(experiment_settings):
    # Extract settings from the dictionary
    search_space = experiment_settings["search_space"]
    num_iterations = experiment_settings["num_iterations"]
    objective_function = experiment_settings["objective_function"]

    # Acquisition and surrogate settings
    acq_settings = experiment_settings["acquisition"]
    surrogate_settings = experiment_settings["surrogate"]

    # Create the data transformer for normalization and standardization
    data_transformer = DataTransformer(search_space, surrogate_settings)

    # Step 1: Generate initial data using Sobol sequences
    X_init, y_init = generate_initial_data(
        objective_function, search_space, n=acq_settings["num_initial_guesses"]
    )

    # Normalize the search space bounds
    lb_normalized = data_transformer.normalize(
        search_space[0].reshape(1, search_space.shape[1])
    )
    ub_normalized = data_transformer.normalize(
        search_space[1].reshape(1, search_space.shape[1])
    )

    # Initialize history with the original (non-transformed) initial data
    X_history = X_init
    y_history = y_init

    # Apply transformations (normalization and standardization)
    X_normalized, y_standardized = data_transformer.apply_transformation(X_init, y_init)

    # Step 2: Initialize the GP model
    rng_key = random.PRNGKey(0)  # JAX random number generator
    gp_model = ExactGP(
        input_dim=search_space.shape[1], kernel=surrogate_settings["kernel"]
    )  # Using specified kernel
    gp_model.fit(rng_key, jnp.array(X_normalized), jnp.array(y_standardized))

    # Step 3: Main loop for Bayesian optimization
    for i in range(num_iterations):
        # Step 3.1: Optimize the acquisition function to find the next point
        X_next_normalized = optimize_acq(
            rng_key,
            gp_model,
            EI,
            num_initial_guesses=acq_settings["num_initial_guesses"],
            lower_bound=lb_normalized,  # In normalized space
            upper_bound=ub_normalized,  # In normalized space
            best_f=jnp.min(jnp.array(y_standardized)),
            maximize=acq_settings["maximize"],
            n=acq_settings["num_samples"],
        )

        X_next_normalized = X_next_normalized.reshape(1, search_space.shape[1])  # Ensure shape is (1, search_space.shape[1])
        
        print()
        print(f'x_next_normalized: {X_next_normalized}')

        # Step 3.2: Inverse-transform the input back to original space
        X_next = data_transformer.inverse_normalize(X_next_normalized)

        print(f'x_next: {X_next}')  
        print()

        # Step 3.3: Evaluate the objective function at the selected point
        y_next = objective_function(X_next)

        # Step 3.4: Add the new (non-transformed) data point to the history
        X_history = np.vstack((X_history, np.array(X_next)))
        y_history = np.vstack((y_history, np.array(y_next)))

        # Apply transformations to the updated dataset (for GP model fitting)
        X_transformed, y_transformed = data_transformer.apply_transformation(X_history, y_history)

        # Step 3.5: Re-train the GP model with the updated transformed dataset
        gp_model.fit(rng_key, jnp.array(X_transformed), jnp.array(y_transformed))

        # Step 3.6: Display progress
        print(f"Iteration {i + 1}: X_next = {X_next}, y_next = {y_next}")

    return X_history, y_history



if __name__ == "__main__":


    # Example experiment settings
    experiment_settings = {
        "search_space": np.array([[-10], [10]]),  # 1D search space example
        "num_iterations": 10,  # Number of optimization iterations
        "objective_function": objective_function,  # Actual objective function
        "acquisition": {  # Acquisition function settings
            "num_samples": 10,  # Number of samples for acquisition function
            "num_initial_guesses": 10,  # Number of initial guesses for acquisition
            "maximize": False,  # Whether to maximize the acquisition function
        },
        "surrogate": {  # Surrogate model (GP) settings
            "kernel": "RBF",  # Kernel for GP model
            "normalize": True,  # Whether to normalize inputs
            "standardize": True,  # Whether to standardize outputs
        },
    }

    # Run the Bayesian optimization
    X_history, y_history = run_bo(experiment_settings)

    # Final result
    print()
    print()
    print()
    print()
    print()
    print("Final X history (in original space):", X_history)
    print("Final y history:", y_history)

    # The final result is in X_history and y_history, containing all evaluated points and their original function values.
    optimal_index = np.argmin(y_history)
    print(f"Optimal X: {X_history[optimal_index]}")
    print(f"Optimal y: {y_history[optimal_index]}")