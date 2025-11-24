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

def _theoretical_gauss_mode_exp(tau, sigma, time_series):
    #this is a theoretical expression for the convolution of a gaussian and an exponential
    # it is a pdf so the integral over all time is 1
    lambda_param = 1 / tau
    if tau/sigma < 0.1:
        #just return a gaussian in this case
        theoretical_gauss_mod_exp = gaussian_distribution(time_series, sigma)
        return theoretical_gauss_mod_exp
       
    theoretical_gauss_mod_exp = (
        lambda_param
        / 2
        * np.exp((lambda_param / 2) * (lambda_param * sigma**2 - 2 * time_series))
        * special.erfc((lambda_param * sigma**2 - time_series) / (np.sqrt(2) * sigma))
    )
    return theoretical_gauss_mod_exp

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

def caclulate_gauss_exp_convolve(tau, sigma, time_series):
    exp_dist = exponential_distribution(time_series, tau)
    gauss_dist = gaussian_distribution(time_series, sigma)
    convolved_distribution = np.convolve(
        exp_dist, gauss_dist, mode="same"
    ) * (np.diff(time_series)[0])  # multiply by the time step to normalize
    return convolved_distribution

def _theoretical_gauss_mode(tau, sigma):
    #this is the mode of the convolution of a gaussian and an exponential
    xmode = (
        0
        - 2**0.5 * sigma * special.erfcinv((np.abs(tau) / sigma) * (2 / np.pi))
        + sigma**2 / tau
    )
    return xmode

def _find_mode_and_max(tau, sigma):
    mean = tau
    variance = sigma**2 + tau**2

    time_series = np.linspace(-(mean+10*np.sqrt(variance)), mean+10*np.sqrt(variance), 20000)
    pdf = _theoretical_gauss_mode_exp(tau, sigma, time_series)
    #check if pdf is all nans
    if np.all(np.isnan(pdf)):
        print("PDF is all NaNs for tau =", tau, "sigma =", sigma)
        print("Using convolution method instead.")
        return np.nan, np.nan
    #check integral
    pdf_integral = np.trapz(pdf, time_series)

    mode_index = np.nanargmax(pdf)
    mode = time_series[mode_index]
    max_value = pdf[mode_index]
    if np.abs(pdf_integral-1)>1e-2:
        print("PDF integral check failed:", pdf_integral)
        return np.nan, np.nan

    return mode, max_value

def get_formed_beam():
    from beam_model import formed
    formed_beam_model = formed.FFTFormedActualBeamModel()
    beam_id_base = np.arange(0,256)
    freqs = np.array([600])
    beam_x = []
    beam_y = []
    for i in range(4):
        beam_ids = beam_id_base + (i*1000)
        beam_positions = formed_beam_model.get_beam_positions(beam_ids, freqs)
        for pos in beam_positions:
            beam_x.append(pos[0][0])
            beam_y.append(pos[0][1])

    # plt.figure()
    # plt.scatter(beam_x, beam_y, alpha=0.5)
    # plt.xlabel('Beam X')
    # plt.ylabel('Beam Y')
    # plt.title('Formed Beam Positions at 600 MHz')
    # plt.show()

    # import pdb; pdb.set_trace()
    return np.min(beam_x), np.max(beam_x), np.min(beam_y), np.max(beam_y)

