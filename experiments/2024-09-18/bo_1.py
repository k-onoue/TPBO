import configparser
import sys
import logging
import warnings

# Load configuration
config = configparser.ConfigParser()
config_path = "./config.ini"
config.read(config_path)
PROJECT_DIR = config["paths"]["project_dir"]
LOG_DIR = config["paths"]["logs_dir"]
sys.path.append(PROJECT_DIR)

import jax.numpy as jnp
import numpy as np
from gpax.acquisition import EI, optimize_acq
from gpax.utils import enable_x64, get_keys

from src.gp import ExactGP
from src.test_functions import sinusoidal_synthetic
from src.utils_bo import DataTransformer, generate_initial_data
from src.utils_experiment import set_logger


# Define the main function for running Bayesian Optimization
def run_bo(experiment_settings):
    logging.info(f"Start BO with settings: {experiment_settings}")

    if experiment_settings["is_x64"]:  # Set to True for higher precision (64-bit)
        enable_x64()

    # Extract settings from the dictionary
    search_space = experiment_settings["search_space"]
    num_iterations = experiment_settings["num_iterations"]
    initial_sample_size = experiment_settings["initial_sample_size"]
    objective_function = experiment_settings["objective_function"]
    seed = experiment_settings["seed"]

    # Acquisition and surrogate settings
    acq_settings = experiment_settings["acquisition"]
    surrogate_settings = experiment_settings["surrogate"]

    # Create the data transformer for normalization and standardization
    data_transformer = DataTransformer(search_space, surrogate_settings)

    # Step 1: Generate initial data using Sobol sequences
    X_init, y_init = generate_initial_data(objective_function, search_space, n=initial_sample_size)
    for x, y in zip(X_init, y_init):
        logging.info(f"X initial: {x}")
        logging.info(f"y initial: {y}")

    # Normalize the search space bounds
    lb_normalized = data_transformer.normalize(search_space[0].reshape(1, search_space.shape[1]))
    ub_normalized = data_transformer.normalize(search_space[1].reshape(1, search_space.shape[1]))

    # Initialize history with the original (non-transformed) initial data
    X_history = X_init
    y_history = y_init

    # Apply transformations (normalization and standardization)
    X_normalized, y_standardized = data_transformer.apply_transformation(X_init, y_init)

    # Step 2: Initialize the GP model
    rng_key_1, rng_key_2 = get_keys(seed)  # Use gpax get_keys for random generation

    # Initialize the GP model with kernel and noise priors
    gp_model = ExactGP(
        input_dim=search_space.shape[1],
        kernel=surrogate_settings["kernel"]
    )
    gp_model.fit(rng_key_1, jnp.array(X_normalized), jnp.array(y_standardized))

    # Step 3: Main loop for Bayesian optimization
    for i in range(num_iterations):
        logging.info(f"Iteration: {i + 1} / {num_iterations}")
        # Step 3.1: Optimize the acquisition function to find the next point
        X_next_normalized = optimize_acq(
            rng_key_2,
            gp_model,
            EI,
            num_initial_guesses=acq_settings["num_initial_guesses"],
            lower_bound=lb_normalized,  # In normalized space
            upper_bound=ub_normalized,  # In normalized space
            best_f=jnp.min(jnp.array(y_standardized)),
            maximize=acq_settings["maximize"],
            n=acq_settings["num_samples"],
            filter_nans=True,
        )

        X_next_normalized = X_next_normalized.reshape(
            1, search_space.shape[1]
        )  # Ensure shape is (1, search_space.shape[1])

        # Step 3.2: Inverse-transform the input back to original space
        X_next = data_transformer.inverse_normalize(X_next_normalized)
        logging.info(f"X new: {X_next}")

        # Step 3.3: Evaluate the objective function at the selected point
        y_next = objective_function(X_next)
        logging.info(f"y new: {y_next}")

        # Step 3.4: Add the new (non-transformed) data point to the history
        X_history = np.vstack((X_history, np.array(X_next)))
        y_history = np.vstack((y_history, np.array(y_next)))

        # Apply transformations to the updated dataset (for GP model fitting)
        X_transformed, y_transformed = data_transformer.apply_transformation(X_history, y_history)

        # Step 3.5: Re-train the GP model with the updated transformed dataset
        gp_model.fit(rng_key_1, jnp.array(X_transformed), jnp.array(y_transformed))

    logging.info("Completed BO loop.")
    return X_history, y_history


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    name = __file__.split("/")[-1].strip(".py")

    # Example experiment settings
    settings = {
        "name": name,  # Experiment name
        "is_x64": False,  # Use 64-bit precision
        "seed": 0,  # Random seed
        "search_space": np.array([[-10], [10]]),  # 1D search space example
        "num_iterations": 100,  # Number of optimization iterations
        "initial_sample_size": 10,  # Number of initial samples
        "objective_function": sinusoidal_synthetic,  # Actual objective function
        "acquisition": {  # Acquisition function settings
            "num_samples": 50,
            "num_initial_guesses": 10,
            "maximize": False,
        },
        "surrogate": {  # Surrogate model (GP) settings
            "kernel": "Matern",  # Automatically Matern52
            "normalize": True,
            "standardize": True,
        },
        "memo": "This is an example experiment for Bayesian optimization.",
    }

    # Set up logging
    set_logger(settings["name"], LOG_DIR)

    # Run the Bayesian optimization
    X_history, y_history = run_bo(settings)

    # Final result
    logging.info("\nFinal X history (in original space):")
    logging.info(f"{X_history}")
    logging.info("Final y history:")
    logging.info(f"{y_history}")

    # The final result is in X_history and y_history, containing all evaluated points and their original function values.
    optimal_index = np.argmin(y_history)
    logging.info(f"Optimal X: {X_history[optimal_index]}")
    logging.info(f"Optimal y: {y_history[optimal_index]}")
