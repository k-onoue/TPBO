import numpy as np


def negate_function(func):
    def negated_func(x):
        return -func(x)

    return negated_func


def sinusoidal_synthetic(x: np.ndarray) -> np.ndarray:
    r"""
    Computes the function f(x) = -(x-1)^2 * \sin(3x + 5^{-1} + 1) for a given numpy input x.

    Args:
        x (np.ndarray): Input array of shape (N, 1) where N is the number of data points.
                        If the input is (N,), it will be automatically reshaped to (N, 1).

    Returns:
        np.ndarray: Output array of shape (N, 1) representing the computed values of f(x).

    f(x) = -(x-1)^2 \sin(3x + 5^{-1} + 1)
    """
    # If the input is of shape (N,), reshape it to (N, 1)
    if x.ndim == 1:
        x = x[:, None]
    elif x.ndim == 2 and x.shape[1] == 1:
        pass
    else:
        raise ValueError("Input must be of shape (N,) or (N, 1)")

    # Compute the function
    term1 = -(x - 1) ** 2
    term2 = np.sin(3 * x + 1 / 5 + 1)
    val = term1 * term2
    return val


def branin_hoo(x: np.ndarray) -> np.ndarray:
    r"""
    Computes the Branin-Hoo function, typically used for benchmarking optimization algorithms.

    Args:
        x (np.ndarray): Input array of shape (N, 2), where N is the number of data points, and
                        each data point contains 2 dimensions [x_1, x_2].

    Returns:
        np.ndarray: Output array of shape (N, 1), representing the computed values of the Branin-Hoo function.

    Raises:
        ValueError: If the input array is not two-dimensional or does not have exactly 2 features per data point.
    """
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError(
            "Input array must be two-dimensional with exactly two features per data point."
        )

    # Extract x1 and x2
    x1 = x[:, 0]
    x2 = x[:, 1]

    pi = np.pi

    # Compute the Branin-Hoo function components
    term1 = (x2 - (5.1 / (4 * pi**2)) * x1**2 + (5 / pi) * x1 - 6) ** 2
    term2 = 10 * (1 - 1 / (8 * pi)) * np.cos(x1)

    # Final value computation and reshaping to (N, 1)
    val = (term1 + term2 + 10).reshape(-1, 1)

    return val


def hartmann6(x: np.ndarray) -> np.ndarray:
    r"""
    Computes the 6-dimensional Hartmann function, typically used for benchmarking optimization algorithms.

    Args:
        x (np.ndarray): Input array of shape (N, 6), where N is the number of data points, and
                        each data point contains 6 dimensions.

    Returns:
        np.ndarray: Output array of shape (N, 1), representing the computed values of the Hartmann-6 function.

    Raises:
        ValueError: If the input array is not two-dimensional or does not have exactly 6 features per data point.
    """
    if x.ndim != 2 or x.shape[1] != 6:
        raise ValueError(
            "Input array must be two-dimensional with exactly six features per data point."
        )

    # Define constants for the Hartmann function
    alpha = np.array([1.00, 1.20, 3.00, 3.20])
    A = np.array(
        [
            [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
            [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
            [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
            [17.0, 8.0, 0.05, 10.0, 0.1, 14.0],
        ]
    )
    P = np.array(
        [
            [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
            [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
            [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
            [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381],
        ]
    )

    # Compute the Hartmann function
    outer_sum = 0
    for i in range(4):
        inner_sum = np.sum(A[i] * (x - P[i]) ** 2, axis=1)
        outer_sum += alpha[i] * np.exp(-inner_sum)

    # Negate the result to match the typical form of the Hartmann-6 function
    val = -outer_sum.reshape(-1, 1)

    return val
