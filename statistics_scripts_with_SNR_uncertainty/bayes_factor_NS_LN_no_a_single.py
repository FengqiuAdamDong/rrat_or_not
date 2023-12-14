#!/usr/bin/env python3
# paths
import sys
import numpy as np
import os
from matplotlib import pyplot as plt
from scipy import optimize as o
import scipy.stats as stats
import dill
import warnings
import argparse
from dynesty.pool import Pool
from dynesty import plotting as dyplot
import dill
import dynesty
import statistics_basic
from scipy.interpolate import RegularGridInterpolator
from dynesty import utils as dyfunc
import glob
import yaml
import cupy as cp
from scipy import special


def read_config(filename, det_snr):
    with open(filename, "r") as file:
        data = yaml.safe_load(file)

    # Extract the sorted items into variables
    detection_curve = data["detection_curve"]
    logn_N_range = data["logn_N_range"]
    logn_mu_range = data["logn_mu_range"]
    logn_std_range = data["logn_std_range"]
    snr_thresh = data["snr_thresh"]
    try:
        flux_cal = data["flux_cal"]
    except:
        flux_cal = 1
    logn_N_range[1] = logn_N_range[1]
    if logn_N_range[0] == -1:
        # change to full range
        logn_N_range[0] = len(det_snr) + 1
    return (
        detection_curve,
        logn_N_range,
        logn_mu_range,
        logn_std_range,
        snr_thresh,
        flux_cal,
    )


def plot_fit_ln(max_mu, max_std, dets, sigma_det):
    fit_x = np.linspace(1e-9, 50, 10000)
    fit_y = statistics.first_plot(fit_x, max_mu, max_std, sigma_det)
    fig, ax = plt.subplots(1, 1)
    ax.hist(
        dets,
        bins="auto",
        density=True,
        label=f"max_mu={max_mu:.2f}, max_std={max_std:.2f}",
    )
    ax.plot(fit_x, fit_y, label="fit")
    ax.set_xlabel("SNR")
    ax.set_ylabel("Probability")
    ax.legend()
    plt.show()


def process_detection_results(real_det):
    with open(real_det, "rb") as inf:
        det_class = dill.load(inf)

    det_fluence = []
    det_width = []
    det_snr = []
    noise_std = []

    for pulse_obj in det_class.sorted_pulses:
        if pulse_obj.det_amp != -1:
            det_fluence.append(pulse_obj.det_fluence)
            det_width.append(pulse_obj.det_std)
            det_snr.append(pulse_obj.det_snr)
            noise_std.append(pulse_obj.noise_std)

    det_fluence = np.array(det_fluence)
    det_width = np.array(det_width)
    det_snr = np.array(det_snr)
    noise_std = np.array(noise_std)

    return det_fluence, det_width, det_snr, noise_std


def plot_detection_results(det_width, det_fluence, det_snr):
    print("mean width", np.mean(det_width))
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].set_title("Widths")
    axs[0, 0].hist(det_width, bins=50)

    axs[1, 0].set_title("Histogram of detected pulses")
    axs[1, 0].hist(det_fluence, bins="auto", density=True)
    axs[1, 0].set_xlabel("FLUENCE")
    axs[1, 1].hist(det_snr, bins="auto", label="snr")
    axs[1, 1].set_title("detected snr")
    axs[1, 1].set_xlabel("snr")
    axs[1, 1].legend()
    axs[0, 1].hist(det_snr, alpha=0.5, bins="auto", label="fit")
    axs[0, 1].set_title(f"detected snr ,ndet {len(det_snr)}")
    axs[0, 1].set_xlabel("snr")
    axs[0, 1].legend()
    plt.show()


def invgammappf(p, a, b):
    boverx = special.gammainccinv(a, p)
    x = b / boverx
    return x


def pt_Uniform_N(x, max_det):
    # need to set conditional prior for mu and sigma
    # lets set the sigma prior to be always between 0 and 2

    # set the prior for mu conditional on sigma
    # min_mu = np.log(max_det/100)
    # max_mu = np.log(max_det)
    # ptmu = (logn_mu_range[1] - logn_mu_range[0]) * x[0] + logn_mu_range[0]
    # ptmu = (max_mu - min_mu) * x[0] + min_mu
    ptmu = stats.norm.ppf(x[0], loc=0, scale=4)
    # ptsigma = (2-0) * x[1] + 0
    ptsigma = stats.invgamma.ppf(x[1], a=1.938)
    # ptsigma = invgammappf(x[1],1,0.5)

    ptN = (logn_N_range[1] - logn_N_range[0]) * x[2] + logn_N_range[0]
    return np.array([ptmu, ptsigma, ptN])


def loglikelihood(theta, det_snr):
    # convert to strict upper limit of the lognorm
    # convert to the standard mu and sigma of a lognorm
    # print("theta",theta)
    a = 0
    lower_c = 0
    upper_c = cp.inf
    # theta = [0,0.5,15000]
    X = {
        "mu": theta[0],
        "std": theta[1],
        "N": theta[2],
        "a": 0,
        "lower_c": lower_c,
        "upper_c": upper_c,
    }
    return statistics.total_p(
        X, snr_arr=det_snr, use_a=False, use_cutoff=True, cuda_device=cuda_device
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate some pulses")
    # add an argument for config file
    parser.add_argument(
        "-i", default="simulated_dir", help="folder with the simulated pulses"
    )
    args = parser.parse_args()
    real_det = args.i

    #####preamble finished#####
    cuda_device = 0

    config_det = real_det.replace(".dill", ".yaml")
    with open(config_det, "r") as inf:
        config = yaml.safe_load(inf)
    det_fluence, det_width, det_snr, noise_std = process_detection_results(real_det)
    print(real_det, config_det)
    (
        detection_curve,
        logn_N_range,
        logn_mu_range,
        logn_std_range,
        snr_thresh,
        flux_cal,
    ) = read_config(config_det, det_snr)
    det_snr = det_snr * flux_cal  # convert to flux units
    detection_curve = config["detection_curve"]
    snr_thresh = statistics_basic.load_detection_fn(
        detection_curve, min_snr_cutoff=snr_thresh, flux_cal=flux_cal
    )
    print("snr_thresh", snr_thresh)
    import statistics

    det_error = statistics.det_error
    # filter the det_snr
    det_snr = det_snr[det_snr > snr_thresh]
    if max(det_snr) > 100:
        print("Warning, the maximum snr is very large, this may cause numerical issues")
    plot_detection_results(det_width, det_fluence, det_snr)
    print(
        "logn_N_range",
        logn_N_range,
        "logn_mu_range",
        logn_mu_range,
        "logn_std_range",
        logn_std_range,
    )
    # load the lookup table
    nDims = 3
    dill_fn = real_det.split("/")[-1]
    dill_fn = dill_fn.split(".")[:-1]
    dill_fn = ".".join(dill_fn)
    checkpoint_fn = f"{dill_fn}_logn.h5"
    print("checkpoint_fn", checkpoint_fn)
    print("starting sampling")
    max_det = np.max(det_snr)
    ln_sampler_a = dynesty.NestedSampler(
        loglikelihood,
        pt_Uniform_N,
        nDims,
        logl_args=[det_snr],
        nlive=128,
        ptform_args=[max_det],
    )
    print("starting run_nested")
    ln_sampler_a.run_nested(checkpoint_file=checkpoint_fn)

    ln_a_sresults = ln_sampler_a.results
    # save the result in a npz file
    np.savez(f"{real_det}_logn_results.npz", results=ln_sampler_a.__dict__)
    fg, ax = dyplot.cornerplot(
        ln_a_sresults,
        color="dodgerblue",
        labels=["mu", "sigma", "N"],
        truths=np.zeros(nDims),
        truth_color="black",
        show_titles=True,
        quantiles=None,
        max_n_ticks=3,
    )
    plt.savefig(f"{real_det}_logn_a_corner.png")
    plt.close()
