import logging
import warnings

import numpy as np
from gpax.acquisition import EI

from _components import run_bo
from _import_from_src import LOG_DIR
from _import_from_src import ExactGP
from _import_from_src import Hartmann6
from _import_from_src import set_logger


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    name = __file__.split("/")[-1].strip(".py")

    # objective_function = SinusoidaSynthetic()
    # objective_function = BraninHoo()
    objective_function = Hartmann6()
    search_space = objective_function.search_space
    is_maximize = objective_function.is_maximize

    # Example experiment settings
    settings = {
        "name": name,  # Experiment name
        "is_x64": False,  # Use 64-bit precision
        "seed": 0,  # Random seed
        "search_space": search_space,  # 1D search space example
        "num_iterations": 100,  # Number of optimization iterations
        "initial_sample_size": 10,  # Number of initial samples
        "objective_function": objective_function,  # Actual objective function
        "acquisition": {  # Acquisition function settings
            "acq_fn_class": EI,  # Acquisition function class
            "num_samples": 50,
            "num_initial_guesses": 10,
            "maximize": is_maximize,
        },
        "surrogate": {  # Surrogate model (GP) settings
            "model_class": ExactGP,  # Model class passed through settings
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
