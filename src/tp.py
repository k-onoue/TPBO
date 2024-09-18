"""
tp.py
=====

Fully Bayesian implementation of Student-t process regression

Revised to match the specified generative process.

Created by [Your Name] (email: your.email@example.com)
"""

import warnings
from typing import Callable, Dict, Optional, Tuple, Type, Union

import jax
import jaxlib
import jax.numpy as jnp
import jax.random as jra
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_median, Predictive

from gpax.kernels import get_kernel
from gpax.utils import split_in_batches

kernel_fn_type = Callable[
    [jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray], jnp.ndarray], jnp.ndarray
]

clear_cache = jax._src.dispatch.xla_primitive_callable.cache_clear


class StudentTProcess:
    """
    Student-t process regression class

    Args:
        input_dim:
            Number of input dimensions
        kernel:
            Kernel function ('RBF', 'Matern', 'Periodic', or custom function)
        mean_fn:
            Optional deterministic mean function (use 'mean_fn_priors' to make it probabilistic)
        kernel_prior:
            Optional custom priors over kernel hyperparameters. Use it when passing your custom kernel.
        mean_fn_prior:
            Optional priors over mean function parameters
        noise_prior_dist:
            Optional custom prior distribution over the observational noise variance.
            Defaults to LogNormal(0,1).
        lengthscale_prior_dist:
            Optional custom prior distribution over kernel lengthscale.
            Defaults to LogNormal(0, 1).
        df_prior_dist:
            Prior distribution over degrees of freedom for the Student-t process.
            Defaults to Exponential(0.1) shifted to ensure df > 2.

    Examples:

        Student-t Process Regression

        >>> # Get random number generator keys for training and prediction
        >>> rng_key, rng_key_predict = gpax.utils.get_keys()
        >>> # Initialize model
        >>> tp_model = gpax.StudentTProcess(input_dim=1, kernel='Matern')
        >>> # Run HMC to obtain posterior samples for the TP model parameters
        >>> tp_model.fit(rng_key, X, y)  # X and y are arrays with dimensions (n, 1) and (n,)
        >>> # Make a prediction on new inputs
        >>> y_pred, y_samples = tp_model.predict(rng_key_predict, X_new)

    """

    def __init__(
        self,
        input_dim: int,
        kernel: Union[str, kernel_fn_type],
        mean_fn: Optional[
            Callable[[jnp.ndarray, Dict[str, jnp.ndarray]], jnp.ndarray]
        ] = None,
        kernel_prior: Optional[Callable[[], Dict[str, jnp.ndarray]]] = None,
        mean_fn_prior: Optional[Callable[[], Dict[str, jnp.ndarray]]] = None,
        noise_prior: Optional[Callable[[], Dict[str, jnp.ndarray]]] = None,
        noise_prior_dist: Optional[dist.Distribution] = None,
        lengthscale_prior_dist: Optional[dist.Distribution] = None,
        df_prior_dist: Optional[dist.Distribution] = None,
    ) -> None:
        clear_cache()
        if noise_prior is not None:
            warnings.warn(
                "`noise_prior` is deprecated and will be removed in a future version. "
                "Please use `noise_prior_dist` instead, which accepts an instance of a "
                "numpyro.distributions Distribution object, e.g., `dist.HalfNormal(scale=0.1)`, "
                "rather than a function that calls `numpyro.sample`.",
                FutureWarning,
            )
        if kernel_prior is not None:
            warnings.warn(
                "`kernel_prior` will remain available for complex priors. However, for "
                "modifying only the lengthscales, it is recommended to use `lengthscale_prior_dist` instead. "
                "`lengthscale_prior_dist` accepts an instance of a numpyro.distributions Distribution object, "
                "e.g., `dist.Gamma(2, 5)`, rather than a function that calls `numpyro.sample`.",
                UserWarning,
            )
        self.kernel_dim = input_dim
        self.kernel = get_kernel(kernel)
        self.kernel_name = kernel if isinstance(kernel, str) else None
        self.mean_fn = mean_fn
        self.kernel_prior = kernel_prior
        self.mean_fn_prior = mean_fn_prior
        self.noise_prior = noise_prior
        self.noise_prior_dist = noise_prior_dist
        self.lengthscale_prior_dist = lengthscale_prior_dist
        self.df_prior_dist = df_prior_dist or dist.Exponential(0.1)
        self.X_train = None
        self.y_train = None
        self.mcmc = None

    def model(self, X: jnp.ndarray, y: jnp.ndarray = None, **kwargs: float) -> None:
        """Student-t process probabilistic model with inputs X and targets y"""
        # Initialize mean function at zeros
        f_loc = jnp.zeros(X.shape[0])

        # Sample kernel parameters
        if self.kernel_prior:
            kernel_params = self.kernel_prior()
        else:
            kernel_params = self._sample_kernel_params()

        # Sample noise
        if self.noise_prior:  # this will be removed in future releases
            noise = self.noise_prior()
        else:
            noise = self._sample_noise()

        # Sample degrees of freedom (nu > 2)
        df = numpyro.sample("df", self.df_prior_dist) + 2.0

        # Add mean function (if any)
        if self.mean_fn is not None:
            args = [X]
            if self.mean_fn_prior is not None:
                args += [self.mean_fn_prior()]
            f_loc += self.mean_fn(*args).squeeze()

        # Compute kernel matrix
        k = self.kernel(X, X, kernel_params, noise, **kwargs)

        # Sample scaling variable r ~ InverseGamma(df / 2, 0.5)
        r = numpyro.sample("r", dist.InverseGamma(df / 2.0, 0.5))

        # Adjust covariance matrix
        cov = k * r * (df - 2)

        # Sample y according to the Multivariate Normal with scaled covariance
        numpyro.sample(
            "y",
            dist.MultivariateNormal(loc=f_loc, covariance_matrix=cov),
            obs=y,
        )

    def fit(
        self,
        rng_key: jnp.array,
        X: jnp.ndarray,
        y: jnp.ndarray,
        num_warmup: int = 2000,
        num_samples: int = 2000,
        num_chains: int = 1,
        chain_method: str = "sequential",
        progress_bar: bool = True,
        print_summary: bool = True,
        device: Type[jaxlib.xla_extension.Device] = None,
        **kwargs: float
    ) -> None:
        """
        Run Hamiltonian Monte Carlo to infer the Student-t process parameters

        Args:
            rng_key: random number generator key
            X: 2D feature vector
            y: 1D target vector
            num_warmup: number of HMC warmup steps
            num_samples: number of HMC samples
            num_chains: number of HMC chains
            chain_method: 'sequential', 'parallel' or 'vectorized'
            progress_bar: show progress bar
            print_summary: print summary at the end of sampling
            device:
                optionally specify a cpu or gpu device on which to run the inference;
                e.g., ``device=jax.devices("cpu")[0]``
            **kwargs:
                Additional arguments passed to the kernel function, such as `jitter`
        """
        X, y = self._set_data(X, y)
        if device:
            X = jax.device_put(X, device)
            y = jax.device_put(y, device)
        self.X_train = X
        self.y_train = y

        init_strategy = init_to_median(num_samples=10)
        kernel = NUTS(self.model, init_strategy=init_strategy)
        self.mcmc = MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            chain_method=chain_method,
            progress_bar=progress_bar,
            jit_model_args=False,
        )
        self.mcmc.run(rng_key, X, y, **kwargs)
        if print_summary:
            self._print_summary()

    def _sample_noise(self) -> jnp.ndarray:
        if self.noise_prior_dist is not None:
            noise_dist = self.noise_prior_dist
        else:
            noise_dist = dist.LogNormal(0, 1)
        return numpyro.sample("noise", noise_dist)

    def _sample_kernel_params(self, output_scale=True) -> Dict[str, jnp.ndarray]:
        """
        Sample kernel parameters with default
        weakly-informative log-normal priors
        """
        if self.lengthscale_prior_dist is not None:
            length_dist = self.lengthscale_prior_dist
        else:
            length_dist = dist.LogNormal(0.0, 1.0)
        with numpyro.plate(
            "ard", self.kernel_dim
        ):  # allows using ARD kernel for kernel_dim > 1
            length = numpyro.sample("k_length", length_dist)
        if output_scale:
            scale = numpyro.sample("k_scale", dist.LogNormal(0.0, 1.0))
        else:
            scale = numpyro.deterministic("k_scale", jnp.array(1.0))
        if self.kernel_name == "Periodic":
            period = numpyro.sample("period", dist.LogNormal(0.0, 1.0))
        kernel_params = {
            "k_length": length,
            "k_scale": scale,
            "period": period if self.kernel_name == "Periodic" else None,
        }
        return kernel_params

    def get_samples(self, chain_dim: bool = False) -> Dict[str, jnp.ndarray]:
        """Get posterior samples (after running the MCMC chains)"""
        return self.mcmc.get_samples(group_by_chain=chain_dim)

    def get_mvn_posterior(
        self,
        X_new: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
        noiseless: bool = False,
        **kwargs: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Returns parameters (mean and cov) of the predictive Multivariate Student-t posterior
        for a single sample of TP parameters
        """
        noise = params["noise"]
        noise_p = noise * (1 - jnp.array(noiseless, int))
        y_residual = self.y_train.copy()
        if self.mean_fn is not None:
            args = [self.X_train, params] if self.mean_fn_prior else [self.X_train]
            y_residual -= self.mean_fn(*args).squeeze()

        # Degrees of freedom
        df = params["df"] + self.X_train.shape[0]

        # Compute kernel matrices
        k_XX = self.kernel(self.X_train, self.X_train, params, noise, **kwargs)
        k_pX = self.kernel(X_new, self.X_train, params, jitter=0.0)
        k_pp = self.kernel(X_new, X_new, params, noise_p, **kwargs)

        # Compute inverse of k_XX
        K_xx_inv = jnp.linalg.inv(k_XX)

        # Compute predictive mean
        mean = jnp.matmul(k_pX, jnp.matmul(K_xx_inv, y_residual))

        # Compute beta
        beta = jnp.dot(y_residual, jnp.matmul(K_xx_inv, y_residual))

        # Compute predictive covariance
        scale = (params["df"] + beta - 2) / (params["df"] + self.X_train.shape[0] - 2)
        cov = k_pp - jnp.matmul(k_pX, jnp.matmul(K_xx_inv, k_pX.T))
        cov = cov * scale

        if self.mean_fn is not None:
            args = [X_new, params] if self.mean_fn_prior else [X_new]
            mean += self.mean_fn(*args).squeeze()

        return mean, cov

    def _predict(
        self,
        rng_key: jnp.ndarray,
        X_new: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
        n: int,
        noiseless: bool = False,
        **kwargs: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Prediction with a single sample of TP parameters"""
        # Get the predictive mean and covariance
        y_mean, cov = self.get_mvn_posterior(X_new, params, noiseless, **kwargs)

        # Compute the Cholesky decomposition of the covariance matrix
        scale_tril = jnp.linalg.cholesky(cov)

        # Draw samples from the predictive Multivariate Student-t distribution
        df = params["df"] + self.X_train.shape[0]

        # # Use numpyro's MultivariateStudentT
        # y_samples = dist.MultivariateStudentT(
        #     df=df, loc=y_mean, covariance_matrix=cov
        # ).sample(rng_key, sample_shape=(n,))

        y_samples = dist.MultivariateStudentT(
            df=df, loc=y_mean, scale_tril=scale_tril
        ).sample(rng_key, sample_shape=(n,))

        return y_mean, y_samples

    def _predict_in_batches(
        self,
        rng_key: jnp.ndarray,
        X_new: jnp.ndarray,
        batch_size: int = 100,
        batch_dim: int = 0,
        samples: Optional[Dict[str, jnp.ndarray]] = None,
        n: int = 1,
        filter_nans: bool = False,
        predict_fn: Callable[[jnp.ndarray, int], Tuple[jnp.ndarray]] = None,
        noiseless: bool = False,
        device: Type[jaxlib.xla_extension.Device] = None,
        **kwargs: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if predict_fn is None:
            predict_fn = lambda xi: self.predict(
                rng_key,
                xi,
                samples,
                n,
                filter_nans,
                noiseless,
                device,
                **kwargs,
            )

        def predict_batch(Xi):
            out1, out2 = predict_fn(Xi)
            out1 = jax.device_put(out1, jax.devices("cpu")[0])
            out2 = jax.device_put(out2, jax.devices("cpu")[0])
            return out1, out2

        y_out1, y_out2 = [], []
        for Xi in split_in_batches(X_new, batch_size, dim=batch_dim):
            out1, out2 = predict_batch(Xi)
            y_out1.append(out1)
            y_out2.append(out2)
        return y_out1, y_out2

    def predict_in_batches(
        self,
        rng_key: jnp.ndarray,
        X_new: jnp.ndarray,
        batch_size: int = 100,
        samples: Optional[Dict[str, jnp.ndarray]] = None,
        n: int = 1,
        filter_nans: bool = False,
        predict_fn: Callable[[jnp.ndarray, int], Tuple[jnp.ndarray]] = None,
        noiseless: bool = False,
        device: Type[jaxlib.xla_extension.Device] = None,
        **kwargs: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Make prediction at X_new with sampled TP parameters
        by splitting the input array into chunks ("batches") and running
        predict_fn (defaults to self.predict) on each of them one-by-one
        to avoid a memory overflow
        """
        y_pred, y_sampled = self._predict_in_batches(
            rng_key,
            X_new,
            batch_size,
            0,
            samples,
            n,
            filter_nans,
            predict_fn,
            noiseless,
            device,
            **kwargs,
        )
        y_pred = jnp.concatenate(y_pred, 0)
        y_sampled = jnp.concatenate(y_sampled, -1)
        return y_pred, y_sampled

    def predict(
        self,
        rng_key: jnp.ndarray,
        X_new: jnp.ndarray,
        samples: Optional[Dict[str, jnp.ndarray]] = None,
        n: int = 1,
        filter_nans: bool = False,
        noiseless: bool = False,
        device: Type[jaxlib.xla_extension.Device] = None,
        **kwargs: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Make prediction at X_new points using posterior samples for TP parameters

        Args:
            rng_key: random number generator key
            X_new: new inputs with *(number of points, number of features)* dimensions
            samples: optional (different) samples with TP parameters
            n: number of samples from Multivariate Student-t posterior for each HMC sample with TP parameters
            filter_nans: filter out samples containing NaN values (if any)
            noiseless:
                Noise-free prediction. It is set to False by default as new/unseen data is assumed
                to follow the same distribution as the training data.
            device:
                optionally specify a cpu or gpu device on which to make a prediction;
                e.g., ```device=jax.devices("gpu")[0]```
            **kwargs:
                Additional arguments passed to the kernel function, such as `jitter`

        Returns:
            Center of the mass of sampled means and all the sampled predictions
        """
        X_new = self._set_data(X_new)
        if samples is None:
            samples = self.get_samples(chain_dim=False)
        if device:
            self._set_training_data(device=device)
            X_new = jax.device_put(X_new, device)
            samples = jax.device_put(samples, device)
        num_samples = len(next(iter(samples.values())))
        vmap_args = (jra.split(rng_key, num_samples), samples)
        predictive = jax.vmap(
            lambda prms: self._predict(prms[0], X_new, prms[1], n, noiseless, **kwargs)
        )
        y_means, y_sampled = predictive(vmap_args)
        if filter_nans:
            y_sampled_ = [y_i for y_i in y_sampled if not jnp.isnan(y_i).any()]
            y_sampled = jnp.array(y_sampled_)
        return y_means.mean(0), y_sampled

    def sample_from_prior(
        self, rng_key: jnp.ndarray, X: jnp.ndarray, num_samples: int = 10
    ):
        """
        Samples from prior predictive distribution at X
        """
        X = self._set_data(X)
        prior_predictive = Predictive(self.model, num_samples=num_samples)
        samples = prior_predictive(rng_key, X)
        return samples["y"]

    def _set_data(
        self, X: jnp.ndarray, y: Optional[jnp.ndarray] = None
    ) -> Union[Tuple[jnp.ndarray], jnp.ndarray]:
        X = X if X.ndim > 1 else X[:, None]
        if y is not None:
            return X, y.squeeze()
        return X

    def _set_training_data(
        self,
        X_train_new: jnp.ndarray = None,
        y_train_new: jnp.ndarray = None,
        device: Type[jaxlib.xla_extension.Device] = None,
    ) -> None:
        X_train = self.X_train if X_train_new is None else X_train_new
        y_train = self.y_train if y_train_new is None else y_train_new
        if device:
            X_train = jax.device_put(X_train, device)
            y_train = jax.device_put(y_train, device)
        self.X_train = X_train
        self.y_train = y_train

    def _print_summary(self):
        samples = self.get_samples(1)
        numpyro.diagnostics.print_summary(samples)




# if __name__ == "__main__":
#     from numpyro.distributions import MultivariateNormal
#     import jax
#     import jax.numpy as jnp

#     # 平均ベクトルと共分散行列を3次元に拡張
#     mean = jnp.zeros((3,))
#     covariance = jnp.array([[1.0, 0.5, 0.3],
#                             [0.5, 1.0, 0.4],
#                             [0.3, 0.4, 1.0]])

#     # MultivariateNormal 分布を定義
#     dist = MultivariateNormal(mean, covariance)

#     # 複数サンプルを取得（10個）
#     samples = dist.sample(jax.random.PRNGKey(1), sample_shape=(10,))
#     print("Multiple Samples:\n", samples)

#     # 各サンプルに対する標準偏差を計算
#     print("axis=0")
#     print(f"shape: {samples.std(axis=0).shape}")
#     print(samples.std(axis=0))
#     print()

#     print("axis=1")
#     print(f"shape: {samples.std(axis=1).shape}")
#     print(samples.std(axis=1))
#     print()

#     print("axis=2")
#     print(f"shape: {samples.std(axis=2).shape}")
#     print(samples.std(axis=2))
#     print()

#     print("axis=-1")
#     print(f"shape: {samples.std(axis=-1).shape}")
#     print(samples.std(axis=-1))






if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    # Generate synthetic data
    np.random.seed(123)
    X = np.linspace(0, 10, 100)[:, None]
    y = np.sin(X) + np.random.normal(0, 0.1, X.shape)

    # Initialize Student-t process regression model
    tp_model = StudentTProcess(input_dim=1, kernel="Matern")

    # Run HMC to obtain posterior samples for the TP model parameters
    rng_key, rng_key_predict = jra.split(jra.PRNGKey(0))
    tp_model.fit(rng_key, X, y)

    # Make a prediction on new inputs
    _, y_sampled = tp_model.predict(rng_key_predict, X, n=1, filter_nans=True)

    
    # print(y_sampled)
    # print(y_sampled.shape)

    print("axis=0")
    print(f"shape: {y_sampled.std(axis=0).shape}")
    print(y_sampled.std(axis=0))
    print()
    print("axis=1")
    print(f"shape: {y_sampled.std(axis=1).shape}")
    print(y_sampled.std(axis=1))
    print()
    print("axis=2")
    print(f"shape: {y_sampled.std(axis=2).shape}")
    print(y_sampled.std(axis=2))
    print()
    print("axis=-1")
    print(f"shape: {y_sampled.std(axis=-1).shape}")
    print(y_sampled.std(axis=-1))
