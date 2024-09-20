from typing import Callable, Type, Tuple

import jax.numpy as jnp

from .gp import ExactGP
from .tp import TP_v2


def switch_to_exactgp_if_needed(
    tp_model: Type[TP_v2], 
    rng_key: jnp.ndarray, 
    X_new: jnp.ndarray, 
    condition: Callable[[float], bool], 
    **kwargs
) -> Tuple[Type[ExactGP], jnp.ndarray, jnp.ndarray]:
    """
    Switches to ExactGP if a condition on TP_v2 is satisfied, and returns predictions using ExactGP.
    
    Args:
        tp_model: The trained TP_v2 model.
        rng_key: Random number generator key.
        X_new: New inputs for prediction.
        condition: A callable that checks the condition on the degrees of freedom.
    
    Returns:
        A tuple of the initialized ExactGP model and its mean and variance predictions.
    """
    # Step 1: Extract the parameters from TP_v2
    samples = tp_model.get_samples(chain_dim=False)
    df = samples["df"].mean()

    # Check the condition based on the degrees of freedom
    if condition(df):
        # Step 2: Transfer parameters to ExactGP (exclude df)
        gp_params = {
            key: value for key, value in samples.items() if key != "df"
        }
        
        # Step 3: Initialize ExactGP with the same parameters
        exactgp_model = ExactGP(
            input_dim=tp_model.kernel_dim,
            kernel=tp_model.kernel_name,  # or pass the kernel function directly
            mean_fn=tp_model.mean_fn,  # same mean function
            kernel_prior=tp_model.kernel_prior,  # transfer priors
            mean_fn_prior=tp_model.mean_fn_prior,
            noise_prior_dist=tp_model.noise_prior_dist,
            lengthscale_prior_dist=tp_model.lengthscale_prior_dist
        )

        # Ensure the training data is set in ExactGP
        exactgp_model.X_train = tp_model.X_train
        exactgp_model.y_train = tp_model.y_train

        # Step 4: Use the transferred parameters to predict using ExactGP
        y_mean, y_var = exactgp_model.get_mvn_posterior(X_new, gp_params, noiseless=True, **kwargs)
        
        return exactgp_model, y_mean, y_var

    # If the condition is not satisfied, continue using TP_v2
    return tp_model, None, None  # Or return the existing predictions from TP_v2 if needed
