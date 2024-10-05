import logging

import jax.numpy as jnp
import numpy as np
from gpax.acquisition import optimize_acq
from gpax.acquisition import UCB, POI, EI
from gpax.utils import enable_x64, get_keys

from _import_from_src import generate_initial_data
from _import_from_src import DataTransformer
from _import_from_src import ExactGP, TP_v2
from _import_from_src import optimize_acq as optimize_acq_proposed
from _import_from_src import get_agt_surrogate
from _import_from_src import UCB_TP, POI_TP, EI_TP
from _import_from_src import add_noise


def set_seed(seed):
    """
    Set the random seed for reproducibility.
    """
    np.random.seed(seed)


# Helper function to initialize data
def initialize_data(
    objective_function, search_space, initial_sample_size, data_transformer, seed
):
    """
    Generate initial data using Sobol sequences and apply transformations.
    """
    X_init, y_init = generate_initial_data(
        objective_function, search_space, n=initial_sample_size, seed=seed
    )
    for x, y in zip(X_init, y_init):
        logging.info(f"X initial: {x}")
        logging.info(f"y initial: {y}")

    # Normalize and standardize the initial data
    X_normalized, y_standardized = data_transformer.apply_transformation(X_init, y_init)

    return X_init, y_init, X_normalized, y_standardized


# Helper function to initialize the surrogate model
def initialize_surrogate_model(
    model_class,
    surrogate_settings,
    X_normalized,
    y_standardized,
    search_space,
    rng_key_1,
    **kwargs,
):
    """
    Initialize the GP model and fit it to the initial data.
    """
    surrogate_model = model_class(
        input_dim=search_space.shape[1], kernel=surrogate_settings["kernel"], **kwargs
    )
    surrogate_model.fit(rng_key_1, jnp.array(X_normalized), jnp.array(y_standardized))
    return surrogate_model


# Helper function to optimize acquisition function
def optimize_acquisition_function(
    rng_key, surrogate_model, acq_settings, lb_normalized, ub_normalized, y_standardized
):
    """
    Optimize the acquisition function to find the next point.
    """
    if lb_normalized.shape[1] != 1:
        lb_normalized = lb_normalized.squeeze()
        ub_normalized = ub_normalized.squeeze()

    X_next_normalized = optimize_acq(
        rng_key,
        surrogate_model,
        acq_settings["acq_fn_class"],
        num_initial_guesses=acq_settings["num_initial_guesses"],
        lower_bound=lb_normalized,
        upper_bound=ub_normalized,
        best_f=jnp.min(jnp.array(y_standardized)),
        maximize=acq_settings["maximize"],
        n=acq_settings["num_samples"],
        filter_nans=True,
    )
    return X_next_normalized


# Helper function to update model with new data
def update_surrogate_model(
    surrogate_model, rng_key_1, X_history, y_history, data_transformer
):
    """
    Update the GP model with new data points.
    """
    X_transformed, y_transformed = data_transformer.apply_transformation(
        X_history, y_history
    )
    surrogate_model.fit(rng_key_1, jnp.array(X_transformed), jnp.array(y_transformed))
    return surrogate_model


# Main Bayesian Optimization function
def run_bo(experiment_settings):
    logging.info(f"Start BO with settings: {experiment_settings}")

    # Step 1: Set precision if required
    if experiment_settings["is_x64"]:
        enable_x64()

    # Step 2: Extract settings
    search_space = experiment_settings["search_space"]
    num_iterations = experiment_settings["num_iterations"]
    initial_sample_size = experiment_settings["initial_sample_size"]
    objective_function = experiment_settings["objective_function"]
    acq_settings = experiment_settings["acquisition"]
    surrogate_settings = experiment_settings["surrogate"]
    model_class = surrogate_settings[
        "model_class"
    ]  # Model class passed through settings
    seed = experiment_settings["seed"]

    noise_strength = experiment_settings.get("noise_strength", 0.0)
    objective_function = add_noise(objective_function, noise_strength)

    set_seed(seed)
    rng_key_1, rng_key_2 = get_keys(seed)

    # Step 3: Initialize data and GP model
    data_transformer = DataTransformer(search_space, surrogate_settings)
    X_init, y_init, X_normalized, y_standardized = initialize_data(
        objective_function, search_space, initial_sample_size, data_transformer, seed
    )
    surrogate_model = initialize_surrogate_model(
        model_class,
        surrogate_settings,
        X_normalized,
        y_standardized,
        search_space,
        rng_key_1,
    )

    logging.info(f"Beta: {float(surrogate_model.get_beta())}")

    # Step 4: Main loop for Bayesian optimization
    X_history, y_history = X_init, y_init
    lb_normalized = data_transformer.normalize(
        search_space[0].reshape(1, search_space.shape[1])
    )
    ub_normalized = data_transformer.normalize(
        search_space[1].reshape(1, search_space.shape[1])
    )

    for i in range(num_iterations):
        logging.info(f"Iteration: {i + 1} / {num_iterations}")

        # Step 4.1: Optimize acquisition function to find next point
        X_next_normalized = optimize_acquisition_function(
            rng_key_2,
            surrogate_model,
            acq_settings,
            lb_normalized,
            ub_normalized,
            y_standardized,
        )
        X_next = data_transformer.inverse_normalize(
            X_next_normalized.reshape(1, search_space.shape[1])
        )
        logging.info(f"X new: {X_next}")

        # Step 4.2: Evaluate the objective function
        y_next = objective_function(X_next)
        logging.info(f"y new: {y_next}")

        # Step 4.3: Update history with the new point
        X_history = np.vstack((X_history, np.array(X_next)))
        y_history = np.vstack((y_history, np.array(y_next)))

        # Step 4.4: Update the GP model with new data
        surrogate_model = update_surrogate_model(
            surrogate_model, rng_key_1, X_history, y_history, data_transformer
        )

        logging.info(f"Beta: {float(surrogate_model.get_beta())}")

    logging.info("Completed BO loop.")
    return X_history, y_history


#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################


# Helper function to optimize acquisition function
def optimize_acquisition_function_proposed(
    rng_key,
    surrogate_model,
    acq_settings,
    lb_normalized,
    ub_normalized,
    y_standardized,
    nu_prime=None,
):
    """
    Optimize the acquisition function to find the next point.
    """
    acq_dict = {
        "GP": {
            "UCB": UCB,
            "POI": POI,
            "EI": EI,
        },
        "TP": {
            "UCB": UCB_TP,
            "POI": POI_TP,
            "EI": EI_TP,
        },
    }

    model_name = "GP" if isinstance(surrogate_model, ExactGP) else "TP"

    if lb_normalized.shape[1] != 1:
        lb_normalized = lb_normalized.squeeze()
        ub_normalized = ub_normalized.squeeze()

    X_next_normalized = optimize_acq_proposed(
        rng_key,
        surrogate_model,
        acq_dict[model_name][acq_settings["acq_fn_class"]],
        num_initial_guesses=acq_settings["num_initial_guesses"],
        lower_bound=lb_normalized,
        upper_bound=ub_normalized,
        best_f=jnp.min(jnp.array(y_standardized)),
        maximize=acq_settings["maximize"],
        n=acq_settings["num_samples"],
        filter_nans=True,
        nu_prime=nu_prime,
    )
    return X_next_normalized


# Main Bayesian Optimization function
def run_bo_proposed(experiment_settings):
    logging.info(f"Start BO with settings: {experiment_settings}")

    # Step 1: Set precision if required
    if experiment_settings["is_x64"]:
        enable_x64()

    # Step 2: Extract settings
    search_space = experiment_settings["search_space"]
    num_iterations = experiment_settings["num_iterations"]
    initial_sample_size = experiment_settings["initial_sample_size"]
    objective_function = experiment_settings["objective_function"]
    acq_settings = experiment_settings["acquisition"]
    surrogate_settings = experiment_settings["surrogate"]
    # model_class = surrogate_settings["model_class"]  # Model class passed through
    seed = experiment_settings["seed"]

    noise_strength = experiment_settings.get("noise_strength", 0.0)
    objective_function = add_noise(objective_function, noise_strength)

    set_seed(seed)
    rng_key_1, rng_key_2 = get_keys(seed)

    # Step 3: Initialize data and GP model
    data_transformer = DataTransformer(search_space, surrogate_settings)
    X_init, y_init, X_normalized, y_standardized = initialize_data(
        objective_function, search_space, initial_sample_size, data_transformer, seed
    )
    tp_model = initialize_surrogate_model(
        TP_v2, surrogate_settings, X_normalized, y_standardized, search_space, rng_key_1
    )

    # Step 4: Main loop for Bayesian optimization
    X_history, y_history = X_init, y_init
    lb_normalized = data_transformer.normalize(
        search_space[0].reshape(1, search_space.shape[1])
    )
    ub_normalized = data_transformer.normalize(
        search_space[1].reshape(1, search_space.shape[1])
    )
    surrogate_model_for_acq, nu_prime = get_agt_surrogate(tp_model)

    for i in range(num_iterations):
        logging.info(f"Iteration: {i + 1} / {num_iterations}")

        # Step 4.1: Optimize acquisition function to find next point
        X_next_normalized = optimize_acquisition_function_proposed(
            rng_key_2,
            surrogate_model_for_acq,
            acq_settings,
            lb_normalized,
            ub_normalized,
            y_standardized,
            nu_prime=nu_prime,
        )
        X_next = data_transformer.inverse_normalize(
            X_next_normalized.reshape(1, search_space.shape[1])
        )
        logging.info(f"X new: {X_next}")

        # Step 4.2: Evaluate the objective function
        y_next = objective_function(X_next)
        logging.info(f"y new: {y_next}")

        # Step 4.3: Update history with the new point
        X_history = np.vstack((X_history, np.array(X_next)))
        y_history = np.vstack((y_history, np.array(y_next)))

        # Step 4.4: Update the GP model with new data
        tp_model = update_surrogate_model(
            tp_model, rng_key_1, X_history, y_history, data_transformer
        )
        surrogate_model_for_acq, nu_prime = get_agt_surrogate(tp_model)

    logging.info("Completed BO loop.")
    return X_history, y_history
