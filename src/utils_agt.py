import logging
from typing import Type, Union

from .gp import ExactGP
from .tp import TP_v2


def get_agt_surrogate(tp_model: Type[TP_v2]) -> Union[Type[ExactGP], Type[TP_v2]]:
    """
    Returns the AGT surrogate model based on the condition on TP_v2.
    """
    X_train = tp_model.X_train
    n = X_train.shape[0]  # X_historyはX_trainに変更
    beta = tp_model.get_beta()
    beta = float(beta)

    criterion = (beta / n) < 1

    logging.info(f"Beta: {beta}")
    logging.info(f"Criterion: {criterion}")

    if criterion:
        gp_model = ExactGP(
            input_dim=tp_model.kernel_dim,
            kernel=tp_model.kernel_name,
            mean_fn=tp_model.mean_fn,
            kernel_prior=tp_model.kernel_prior,
            mean_fn_prior=tp_model.mean_fn_prior,
            noise_prior_dist=tp_model.noise_prior_dist,
            lengthscale_prior_dist=tp_model.lengthscale_prior_dist,
        )

        gp_model.X_train = X_train
        gp_model.y_train = tp_model.y_train 

        gp_model.mcmc = tp_model.mcmc

        return gp_model, None # nu_prime 
    else:
        return tp_model, 2.0 # nu_prime 
