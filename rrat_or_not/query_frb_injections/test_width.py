import numpy as np
from matplotlib import pyplot as plt
from scipy import special
import numpy as np


def exponential_distribution(t, tau):
    """Generate samples from an exponential distribution.

    Parameters:
    t (array-like): Array of time values.
    Returns:
    probabilities (array-like): Corresponding probabilities from the exponential distribution.
    """
    t = np.array(t)
    probabilities = (1 / tau) * np.exp(-t / tau)
    probabilities[t < 0] = 0
    return probabilities


def gaussian_distribution(t, sigma):
    """Generate samples from a Gaussian distribution.

    Parameters:
    t (array-like): Array of time values.
    sigma (float): Standard deviation of the Gaussian distribution.
    Returns:
    probabilities (array-like): Corresponding probabilities from the Gaussian distribution.
    """
    t = np.array(t)
    coeff = 1 / (sigma * np.sqrt(2 * np.pi))
    exponent = -0.5 * (t / sigma) ** 2
    probabilities = coeff * np.exp(exponent)
    return probabilities

import sys
tau = float(sys.argv[1])  # e.g., tau = 5
sigma = float(sys.argv[2])  # e.g., sigma = 2
time_series = np.linspace(-1000, 1000, 100000)  # Time values from -10 to 50 ms
exp_distribution = exponential_distribution(
    time_series, tau=tau
)  # Exponential with tau=5 ms
h = 1
gauss_distribution =h * gaussian_distribution(
    time_series, sigma=sigma
)  # Gaussian with sigma=2 ms


from utils import _theoretical_gauss_mode_exp
from utils import _theoretical_gauss_mode
theoretical_gauss_mod_exp = _theoretical_gauss_mode_exp(tau, sigma, time_series)
theoretical_gauss_mod_exp[np.isnan(theoretical_gauss_mod_exp)] = 0
mean = np.trapz( time_series * theoretical_gauss_mod_exp, time_series)
variance = np.trapz( time_series**2 * theoretical_gauss_mod_exp, time_series) - mean**2
print(mean)
print(variance)
print("theoretical var:" ,sigma**2 + tau**2)
xmode = _theoretical_gauss_mode(tau, sigma)
mode_height = _theoretical_gauss_mode_exp(tau, sigma, xmode)
effective_width = 1/mode_height

print(np.trapezoid(theoretical_gauss_mod_exp, time_series))
theoretical_gauss_mod_exp[np.isnan(theoretical_gauss_mod_exp)] = 0
# convolve the two
convolved_distribution = np.convolve(
    exp_distribution, gauss_distribution, mode="same"
) * (
    np.diff(time_series)[0]
)  # multiply by the time step to normalize
# integrate the convolved distribution
print(np.trapezoid(convolved_distribution, time_series))
from matplotlib import pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(
    time_series,
    exp_distribution,
    label="Exponential Distribution (tau=5 ms)",
    color="blue",
)
plt.plot(
    time_series,
    gauss_distribution,
    label="Gaussian Distribution (sigma=2 ms)",
    color="orange",
)
plt.plot(
    time_series, convolved_distribution, label="Convolved Distribution", color="green"
)
plt.plot(
    time_series,
    theoretical_gauss_mod_exp,
    label="Theoretical Gaussian Modified Exponential",
    color="purple",
    linestyle=":",
)
plt.axvline(xmode, color="red", linestyle="--", label="Theoretical Mode")
plt.axhline(mode_height, color="gray", linestyle="--", label="Theoretical Mode Height")
plt.yscale("log")
plt.xlabel("Time (ms)")
plt.ylabel("Probability Density")
plt.title("Exponential and Gaussian Distributions with Convolution")
plt.legend()
plt.show()
