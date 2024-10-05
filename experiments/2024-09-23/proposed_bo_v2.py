import logging
import os
import warnings
import argparse
import numpy as np

from _components import run_bo_proposed_v2
from _import_from_src import LOG_DIR
from _import_from_src import set_logger
from _import_from_src import SinusoidalSynthetic, BraninHoo, Hartmann6


objective_dict = {
    "SinusoidalSynthetic": SinusoidalSynthetic,
    "BraninHoo": BraninHoo,
    "Hartmann6": Hartmann6,
}

acquisition_list = ["UCB", "POI", "EI"]


def parse_args():
    parser = argparse.ArgumentParser(description="Bayesian Optimization Experiment")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility. Default is 0.",
    )
    parser.add_argument(
        "--objective",
        choices=objective_dict.keys(),
        default="SinusoidalSynthetic",
        help="Objective function for optimization",
    )
    parser.add_argument(
        "--noise_strength",
        type=float,
        default=0,
        help="Noise strength for objective function",
    )
    parser.add_argument(
        "--acquisition",
        choices=acquisition_list,
        default="UCB",
        help="Acquisition function for Bayesian optimization",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of optimization iterations",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Ignore specific warnings (if known) instead of all warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    name = os.path.splitext(__file__.split("/")[-1])[0]

    # Parse arguments
    args = parse_args()

    # set random seed
    seed = args.seed

    # Set objective function and acquisition function based on user input
    objective_function = objective_dict[args.objective]()
    acquisition_function = args.acquisition

    search_space = objective_function.search_space
    is_maximize = objective_function.is_maximize

    # Example experiment settings
    settings = {
        "name": f"{name}_{args.objective}_AGTv2_{args.acquisition}_noise{args.noise_strength}_seed{seed}",  # Experiment name
        "is_x64": False,  # Use 64-bit precision
        "seed": seed,  # Random seed
        "search_space": search_space,  # 1D search space example
        "num_iterations": args.iterations,  # Number of optimization iterations
        "initial_sample_size": 5,  # Number of initial samples
        "objective_function": objective_function,  # Actual objective function
        "noise_strength": args.noise_strength,  # Noise strength for objective function
        "acquisition": {  # Acquisition function settings
            "acq_fn_class": args.acquisition,  # Acquisition function class
            "num_samples": 1,
            "num_initial_guesses": 10,
            "maximize": is_maximize,
        },
        "surrogate": {  # Surrogate model (GP or TP) settings
            "model_class": "AGTv2",
            "kernel": "Matern",  # Automatically Matern52
            "normalize": True,
            "standardize": True,
        },
        "memo": f"Experiment using {args.objective} with {args.acquisition} acquisition.",
    }

    # Set up logging
    set_logger(settings["name"], LOG_DIR)

    # Run the Bayesian optimization
    X_history, y_history = run_bo_proposed_v2(settings)

    # Final result
    logging.info("\nFinal X history (in original space):")
    logging.info(f"{X_history}")
    logging.info("Final y history:")
    logging.info(f"{y_history}")

    # The final result is in X_history and y_history, containing all evaluated points and their original function values.
    if is_maximize:
        optimal_index = np.argmax(y_history)

    logging.info(f"Optimal X: {X_history[optimal_index]}")
    logging.info(f"Optimal y: {y_history[optimal_index]}")
