"""
==============

Base acquisition functions for Student-t process surrogate models.
"""

from typing import Type, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as onp
import numpyro.distributions as dist
from gpax.acquisition.acquisition import _compute_penalties
from gpax.utils import get_keys

# from .tp import TP_v1 as TP
from .tp import TP_v2 as TP


def _compute_mean_and_var_tp(
    rng_key: jnp.ndarray, model: Type[TP], X: jnp.ndarray, n: int, noiseless: bool, **kwargs
) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
    """
    Computes predictive mean, variance, and degrees of freedom for Student-t process.
    """
    if model.mcmc is not None:
        _, y_sampled = model.predict(rng_key, X, n=n, noiseless=noiseless, **kwargs)
        y_sampled = y_sampled.reshape(n * y_sampled.shape[0], -1)
        mean, var = y_sampled.mean(0), y_sampled.var(0)
        df = model.get_samples()["df"].mean()  # Get the mean degrees of freedom
    else:
        mean, var, df = model.get_mvt_posterior(X, model.get_samples(chain_dim=False), **kwargs)

    return mean, var, df


def UCB_TP(rng_key: jnp.ndarray, model: Type[TP],
           X: jnp.ndarray, beta: float = .25,
           maximize: bool = False, n: int = 1,
           noiseless: bool = False,
           penalty: Optional[str] = None,
           recent_points: jnp.ndarray = None,
           grid_indices: jnp.ndarray = None,
           penalty_factor: float = 1.0,
           **kwargs) -> jnp.ndarray:
    r"""
    Upper confidence bound for Student-t Process

    Given a probabilistic model :math:`m` that models the objective function :math:`f`,
    the Upper Confidence Bound for a Student-t process at an input point :math:`x` is defined as:

    .. math::

        UCB(x) = \mu(x) + \kappa \cdot \sigma(x)

    where:
    - :math:`\mu(x)` is the predictive mean,
    - :math:`\sigma(x)` is the predictive standard deviation scaled by the degrees of freedom,
    - :math:`\kappa` is the exploration-exploitation trade-off parameter.

    Args:
        rng_key: JAX random number generator key
        model: trained Student-t process model
        X: new inputs
        beta: coefficient balancing exploration-exploitation trade-off
        maximize: If True, assumes that BO is solving maximization problem
        n: number of samples drawn from each predictive distribution
        noiseless:
            If True, performs noise-free prediction. Defaults to False, meaning that it assumes
            the new/unseen data follows the same distribution as the training data.
        penalty:
            Optional penalty applied to discourage evaluation near recent points.
        recent_points: An array of recently visited points.
        grid_indices: Grid indices of data points in X array for the penalty term calculation.
        penalty_factor: Penalty factor applied to acquisition values.
        **kwargs: Additional arguments passed to the kernel function, such as `jitter`.

    Returns:
        Acquisition function values.
    """

    if penalty and not isinstance(recent_points, (onp.ndarray, jnp.ndarray)):
        raise ValueError("Please provide an array of recently visited points")

    X = X[:, None] if X.ndim < 2 else X

    # Compute predictive mean and variance (moments) for the Student-t process
    mean, var, df = _compute_mean_and_var_tp(rng_key, model, X, n, noiseless, **kwargs)

    def ucb_tp(moments: Tuple[jnp.ndarray, jnp.ndarray], beta: float, df: float, maximize: bool = False) -> jnp.ndarray:
        """
        Inner function for computing UCB with Student-t Process adjustments

        The UCB for a Student-t process at an input point :math:`x` is defined as:

        .. math::

            UCB(x) = \mu(x) + \kappa \cdot \sigma(x)

        where:
        - :math:`\mu(x)` is the predictive mean.
        - :math:`\sigma(x)` is the predictive standard deviation scaled by the degrees of freedom.
        - :math:`\kappa` is the exploration-exploitation trade-off parameter.

        Args:
            moments:
                Tuple with predictive mean and variance (first and second moments of predictive distribution).
            beta: coefficient balancing exploration-exploitation trade-off
            df: degrees of freedom from the Student-t process
            maximize: If True, assumes that BO is solving maximization problem

        Returns:
            UCB acquisition function values.
        """
        mean, var = moments
        # Scale variance by degrees of freedom (Student-t scaling)
        delta = jnp.sqrt(beta * var * (df / (df - 2)))

        if maximize:
            return mean + delta
        else:
            return -(mean - delta)  # return a negative acq for argmax in BO

    # Compute UCB with Student-t process adjustments (variance scaled by df)
    acq = ucb_tp((mean, var), beta, df, maximize)

    if penalty:
        acq -= _compute_penalties(X, recent_points, penalty, penalty_factor, grid_indices)

    return acq


def EI_TP(rng_key: jnp.ndarray, model: Type[TP],
          X: jnp.ndarray, best_f: float = None,
          maximize: bool = False, n: int = 1,
          noiseless: bool = False,
          penalty: Optional[str] = None,
          recent_points: jnp.ndarray = None,
          grid_indices: jnp.ndarray = None,
          penalty_factor: float = 1.0,
          **kwargs) -> jnp.ndarray:
    r"""
    Student-t Process Expected Improvement (EI)

    This function implements the Expected Improvement acquisition function 
    for a Student-t process surrogate model.

    Args:
        rng_key: JAX random number generator key
        model: trained Student-t process model
        X: new inputs
        best_f: Best function value observed so far.
        maximize: If True, assumes that BO is solving maximization problem.
        n: number of samples drawn from each predictive distribution.
        noiseless: Noise-free prediction. Defaults to False.
        penalty: Optional penalty applied to discourage re-evaluation near recent points.
        recent_points: An array of recently visited points.
        grid_indices: Grid indices of data points in X array for penalty term calculation.
        penalty_factor: Penalty factor applied to acquisition values.
        **kwargs: Additional arguments passed to the kernel function.

    Returns:
        Acquisition function values.
    """
    if penalty and not isinstance(recent_points, (onp.ndarray, jnp.ndarray)):
        raise ValueError("Please provide an array of recently visited points")

    X = X[:, None] if X.ndim < 2 else X

    # Compute predictive mean, variance, and degrees of freedom from the TP model
    mean, var, df = _compute_mean_and_var_tp(rng_key, model, X, n, noiseless, **kwargs)

    def ei_tp(moments: Tuple[jnp.ndarray, jnp.ndarray, float],
              best_f: float = None,
              maximize: bool = False,
              **kwargs) -> jnp.ndarray:
        r"""
        Inner function for computing Expected Improvement (EI) for a Student-t Process

        EI(x) = (\hat{y} - \mu) \Phi_s(z) + \frac{\nu}{\nu - 1} \left( 1 + \frac{z^2}{\nu} \right) \sigma \phi_s(z)

        where:
        - \(\mu(x)\) is the predictive mean.
        - \(\sigma(x)\) is the predictive standard deviation.
        - \(\nu\) is the degrees of freedom from the Student-t process.
        - \(\hat{y}\) is the best known function value.

        Args:
            moments: Tuple containing the predictive mean, variance, and degrees of freedom (df).
            best_f: Best function value observed so far.
            maximize: If True, assumes that BO is solving maximization problem.
        
        Returns:
            Expected Improvement (EI) acquisition function values.
        """
        mean, var, df = moments
        sigma = jnp.sqrt(var)
        
        if best_f is None:
            best_f = mean.max() if maximize else mean.min()
        
        # Compute z = (best_f - mean) / sigma
        z = (best_f - mean) / sigma
        if maximize:
            z = -z
        
        # Standard Student-t distribution CDF and PDF
        student_t = dist.StudentT(df)
        phi_s = jnp.exp(student_t.log_prob(z))  # PDF
        Phi_s = student_t.cdf(z)                # CDF

        # Compute EI using the Student-t process formula
        ei_value = (best_f - mean) * Phi_s + (df / (df - 1)) * (1 + (z ** 2) / df) * sigma * phi_s
        
        return ei_value

    # Compute EI for Student-t process
    acq = ei_tp((mean, var, df), best_f, maximize)

    if penalty:
        acq -= _compute_penalties(X, recent_points, penalty, penalty_factor, grid_indices)

    return acq









