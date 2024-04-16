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
from statistics import statistics_ln
from bayes_factor_LNLN import read_config
from bayes_factor_LNLN import process_detection_results
from bayes_factor_LNLN import plot_detection_results
from bayes_factor_LNLN import load_selection_effects


def pt_Uniform_N(x, logn_N_range):
    # need to set conditional prior for mu and sigma
    # lets set the sigma prior to be always between 0 and 2
    ptmu = stats.norm.ppf(x[0], loc=0, scale=4)
    ptmu_w = stats.norm.ppf(x[2], loc=-4.6, scale=3)
    ptsigma = stats.invgamma.ppf(x[1], a=1.938)
    ptsigma_w = stats.invgamma.ppf(x[3], a=1.938)
    ptN = stats.randint.ppf(x[4], logn_N_range[0], logn_N_range[1])
    return np.array([ptmu, ptsigma, ptmu_w, ptsigma_w, ptN])


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
        "std": theta[1],
        "mu_w": theta[2],
        "std_w": theta[3],
        "N": theta[4],
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
        amp_dist="ln",
        w_dist="ln",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate some pulses")
    # add an argument for config file
    parser.add_argument(
        "-y",
        default="yaml file",
        help="yaml file with the detection curve and other parameters",
    )
    args = parser.parse_args()

    #####preamble finished#####
    cuda_device = 0

    config_det = args.y
    det_snr = np.array([])
    det_width = np.array([])
    # if the width is very narrow use the low width flag
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
        snr_upper,
        width_upper,
    ) = read_config(config_det)
    low_width_flag = False
    likelihood_calc, det_snr, det_width = load_selection_effects(
        detection_curve,
        snr_thresh=snr_thresh,
        width_thresh=width_thresh,
        det_snr=det_snr,
        det_width=det_width,
        low_width_flag=low_width_flag,
        snr_upper=snr_upper,
        width_upper=width_upper,
    )

    if logn_N_range[0] == -1:
        logn_N_range[0] = 0

    nDims = 5

    dill_fn = config_det.split("/")[-1]
    dill_fn = dill_fn.split(".")[:-1]
    dill_fn = ".".join(dill_fn)
    checkpoint_fn = f"{dill_fn}_lnln.h5"
    print("checkpoint_fn", checkpoint_fn)

    print("starting sampling")
    ln_sampler_a = dynesty.NestedSampler(
        loglikelihood,
        pt_Uniform_N,
        nDims,
        logl_args=[det_snr, det_width, likelihood_calc, low_width_flag],
        nlive=1024,
        ptform_args=[logn_N_range],
    )
    print("starting run_nested")
    ln_sampler_a.run_nested(checkpoint_file=checkpoint_fn)

    ln_a_sresults = ln_sampler_a.results
    # save the result in a npz file
    np.savez(f"{config_det}_lnln_results.npz", results=ln_sampler_a.results)
    fg, ax = dyplot.cornerplot(
        ln_a_sresults,
        color="dodgerblue",
        labels=["mu", "sigma", "mu_w", "std_w", "N"],
        truths=np.zeros(nDims),
        truth_color="black",
        show_titles=True,
        quantiles=None,
        max_n_ticks=3,
    )
    plt.savefig(f"{dill_fn}_lnln_corner.png")
    plt.close()
