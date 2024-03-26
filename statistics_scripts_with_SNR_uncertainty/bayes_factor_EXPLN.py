#!/usr/bin/env python3
# paths
import sys
import numpy as np
import os
from matplotlib import pyplot as plt
from scipy import optimize as o
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
import scipy.stats as stats
from bayes_factor_LNLN import read_config
from bayes_factor_LNLN import process_detection_results
from bayes_factor_LNLN import plot_detection_results


def pt_Uniform_N(x, max_det, max_width, logn_N_range):
    # need to set conditional prior for mu and sigma
    # lets set the sigma prior to be always between 0 and 2

    ptk = stats.truncnorm.ppf(x[0], a=-1, b= np.inf, loc=1, scale=1)
    ptmu_w = stats.norm.ppf(x[1],loc=-4.6, scale=3)
    ptsigma_w = stats.invgamma.ppf(x[2], a=1.938)
    ptN = stats.randint.ppf(x[3], logn_N_range[0], logn_N_range[1])

    return np.array([ptk, ptmu_w, ptsigma_w, ptN])


def loglikelihood(theta, det_snr, det_width, likelihood_calc, low_width_flag):
    # convert to strict upper limit of the lognorm
    # convert to the standard mu and sigma of a lognorm
    # print("theta",theta)
    a = 0
    lower_c = 0
    # mean,var = mu_std_to_mean_var(theta[0],theta[1])
    # median = np.exp(theta[0])
    # upper_c = median * 50
    upper_c = cp.inf
    # xlim=100
    # theta = [0,0.75,-5.3,0.1,161787]
    X = {
        "mu": theta[0],
        "std": 0,
        "mu_w": theta[1],
        "std_w": theta[2],
        "N": theta[3],
        "a": 0,
        "lower_c": lower_c,
        "upper_c": upper_c,
    }
    return likelihood_calc.total_p_cupy(
        X,
        snr_arr=det_snr,
        width_arr=det_width,
        use_a=False,
        use_cutoff=True,
        cuda_device=cuda_device,
        low_width=low_width_flag,
        amp_dist="exp",
        w_dist="ln",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate some pulses")
    # add an argument for config file
    parser.add_argument(
        "-i", default="simulated_dir", help="folder with the simulated pulses"
    )
    args = parser.parse_args()
    real_det = args.i
    from statistics import statistics_ln

    #####preamble finished#####
    cuda_device = 0

    config_det = real_det.replace(".dill", ".yaml")

    #if the width is very narrow use the low width flag
    (
        detection_curve,
        logn_N_range,
        logn_mu_range,
        logn_std_range,
        logn_mu_w_range,
        logn_std_w_range,
        snr_thresh,
        width_thresh,
        flux_cal,
    ) = read_config(config_det)

    #load the selection effects
    likelihood_calc = statistics_ln(
        detection_curve,
        plot=True,
        flux_cal=flux_cal,
        snr_cutoff=snr_thresh,
        width_cutoff=width_thresh,
    )

    det_fluence, det_width, det_snr, noise_std, likelihood_calc, low_width_flag, logN_lower = process_detection_results(real_det, snr_thresh, width_thresh, likelihood_calc)
    if logn_N_range[0] == -1:
        logn_N_range[0] = logN_lower

    # remove_mask = (det_snr < 6)&(det_width < 6e-3)
    # det_snr = det_snr[~remove_mask]
    # det_width = det_width[~remove_mask]

    print(f"number of detections {len(det_snr)}")
    plot_detection_results(det_width, det_fluence, det_snr)

    nDims = 4

    dill_fn = real_det.split("/")[-1]
    dill_fn = dill_fn.split(".")[:-1]
    dill_fn = ".".join(dill_fn)
    checkpoint_fn = f"{dill_fn}_expln.h5"
    print("checkpoint_fn", checkpoint_fn)

    print("starting sampling")
    max_det = np.max(det_snr)
    max_width = np.max(det_width)
    ln_sampler_a = dynesty.NestedSampler(
        loglikelihood,
        pt_Uniform_N,
        nDims,
        logl_args=[det_snr, det_width, likelihood_calc, low_width_flag],
        nlive=256,
        ptform_args=[max_det, max_width, logn_N_range],
    )
    print("starting run_nested")
    ln_sampler_a.run_nested(checkpoint_file=checkpoint_fn)

    ln_a_sresults = ln_sampler_a.results
    # save the result in a npz file
    np.savez(f"{real_det}_expln_results.npz", results=ln_sampler_a.results)
    fg, ax = dyplot.cornerplot(
        ln_a_sresults,
        color="dodgerblue",
        labels=["k", "mu_w", "std_w", "N"],
        truths=np.zeros(nDims),
        truth_color="black",
        show_titles=True,
        quantiles=None,
        max_n_ticks=3,
    )
    plt.savefig(f"{real_det}_expln_corner.png")
    plt.close()
