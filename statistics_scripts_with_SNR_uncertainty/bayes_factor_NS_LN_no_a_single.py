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


def read_config(filename, det_snr):
    with open(filename, "r") as file:
        data = yaml.safe_load(file)

    # Extract the sorted items into variables
    detection_curve = data["detection_curve"]
    logn_N_range = data["logn_N_range"]
    logn_mu_range = data["logn_mu_range"]
    logn_std_range = data["logn_std_range"]
    logn_mu_w_range = data["logn_mu_w_range"]
    logn_std_w_range = data["logn_std_w_range"]

    snr_thresh = data["snr_thresh"]
    width_thresh = data["width_thresh"]
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
        logn_mu_w_range,
        logn_std_w_range,
        snr_thresh,
        width_thresh,
        flux_cal,
    )

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
    fig, axs = plt.subplots(1, 2)
    axs[0].set_title("Widths")
    axs[0].hist(det_width, bins=50)
    axs[0].set_xlabel("Width (s)")
    axs[0].set_ylabel("Counts")

    axs[1].hist(det_snr, bins=50)
    axs[1].set_title(f"Detected Pulse S/N")
    axs[1].set_xlabel("S/N")
    plt.savefig("detection_results.pdf")
    plt.tight_layout()
    plt.show()


def pt_Uniform_N(x, max_det, max_width, logn_N_range):
    # need to set conditional prior for mu and sigma
    # lets set the sigma prior to be always between 0 and 2
    #
    # ptsigma = (logn_std_range[1] - logn_std_range[0]) * x[1] + logn_std_range[0]

    # ptsigma = (2**x[0]) / (1e-2**(x[0]-1))

    # set the prior for mu conditional on sigma
    # min_mu = np.log(max_det / 100)
    # max_mu = np.log(max_det)
    # ptmu = (logn_mu_range[1] - logn_mu_range[0]) * x[0] + logn_mu_range[0]
    # min_mu = 0.99
    # max_mu = 1.01
    # ptmu = (max_mu - min_mu) * x[0] + min_mu
    ptmu = stats.norm.ppf(x[0],loc=0, scale=4)

    # ptN = (logn_N_range[1] - logn_N_range[0]) * x[2] + logn_N_range[0]
    # min_mu_w = np.log(max_width / 100)
    # max_mu_w = np.log(max_width)
    # min_mu_w = -4.299
    # max_mu_w = -4.301

    # ptmu_w = (max_mu_w - min_mu_w) * x[2] + min_mu_w
    ptmu_w = stats.norm.ppf(x[2],loc=-4.6, scale=3)

    # min_pt_sigma = 0.19
    # max_pt_sigma = 0.21
    # min_pt_sigma_w = -6.215
    # max_pt_sigma_w = -6.213
    # min_pt_sigma = 0.7499
    # max_pt_sigma = 0.7501
    # min_pt_sigma_w = 0.01
    # max_pt_sigma_w = 1.0
    # min_pt_sigma = 0.01
    # max_pt_sigma = 2.0
    # min_pt_sigma_w = 0.01
    # max_pt_sigma_w = 1.0
    # ptsigma = (max_pt_sigma - min_pt_sigma) * x[1] + min_pt_sigma
    # ptsigma_w = (max_pt_sigma_w - min_pt_sigma_w) * x[3] + min_pt_sigma_w
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
    with open(config_det, "r") as inf:
        config = yaml.safe_load(inf)

    # for real_det,config_det in zip(dill_files,config_files):
    # check if png already made
    # png_fp = f"{real_det}_logn_a_corner.png"
    # if os.path.exists(png_fp):
    #    print(f"skipping {png_fp}")
    #    sys.exit(1)
    det_fluence, det_width, det_snr, noise_std = process_detection_results(real_det)
    #if the width is very narrow use the low width flag
    low_width_flag = np.mean(det_width) < 4e-3
    low_width_flag = False
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
    ) = read_config(config_det, det_snr)
    det_snr = det_snr * flux_cal  # convert to flux units
    detection_curve = config["detection_curve"]
    likelihood_calc = statistics_ln(
        detection_curve,
        plot=True,
        flux_cal=flux_cal,
        snr_cutoff=snr_thresh,
        width_cutoff=width_thresh,
    )
    likelihood_calc.convolve_p_detect(low_width=low_width_flag)

    print("snr_thresh", snr_thresh)
    print("width_thresh", width_thresh)
    width_wide_thresh = 28e-3
    print("width_wide_thresh", width_wide_thresh)
    # filter the det_snr
    mask = (det_snr > snr_thresh) & (det_width > width_thresh) & (det_width < width_wide_thresh)
    det_snr = det_snr[mask]
    det_width = det_width[mask]

    # remove_mask = (det_snr < 2.8)&(det_width < 5e-3)
    # det_snr = det_snr[~remove_mask]
    # det_width = det_width[~remove_mask]

    likelihood_calc.calculate_pdet(det_snr,det_width)
    print(f"number of detections {len(det_snr)}")
    plot_detection_results(det_width, det_fluence, det_snr)

    nDims = 5

    dill_fn = real_det.split("/")[-1]
    dill_fn = dill_fn.split(".")[:-1]
    dill_fn = ".".join(dill_fn)
    checkpoint_fn = f"{dill_fn}_logn.h5"
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
    np.savez(f"{real_det}_logn_results.npz", results=ln_sampler_a.__dict__)
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
    plt.savefig(f"{real_det}_logn_a_corner.png")
    plt.close()
