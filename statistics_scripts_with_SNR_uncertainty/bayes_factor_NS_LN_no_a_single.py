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
    # plt.figure()
    # plt.scatter(np.log(det_width), np.log(det_snr), alpha=0.5,s=1)
    # plt.xlabel("log width")
    # plt.ylabel("log snr")
    plt.show()


def pt_Uniform_N(x, max_det, max_width, logn_N_range):
    # need to set conditional prior for mu and sigma
    # lets set the sigma prior to be always between 0 and 2
    #
    # ptsigma = (logn_std_range[1] - logn_std_range[0]) * x[1] + logn_std_range[0]

    # ptsigma = (2**x[0]) / (1e-2**(x[0]-1))

    # set the prior for mu conditional on sigma
    min_mu = np.log(max_det / 100)
    max_mu = np.log(max_det)
    # ptmu = (logn_mu_range[1] - logn_mu_range[0]) * x[0] + logn_mu_range[0]
    # min_mu = 0.99
    # max_mu = 1.01
    ptmu = (max_mu - min_mu) * x[0] + min_mu
    # ptN = (logn_N_range[1] - logn_N_range[0]) * x[2] + logn_N_range[0]
    min_mu_w = -20
    max_mu_w = np.log(max_width)
    # min_mu_w = -4.299
    # max_mu_w = -4.301

    ptmu_w = (max_mu_w - min_mu_w) * x[2] + min_mu_w

    # min_pt_sigma = 0.19
    # max_pt_sigma = 0.21
    # min_pt_sigma_w = 0.29
    # max_pt_sigma_w = 0.31
    # min_pt_sigma = 0.7499
    # max_pt_sigma = 0.7501
    # min_pt_sigma_w = 0.099
    # max_pt_sigma_w = 0.101
    min_pt_sigma = 0.01
    max_pt_sigma = 2.0
    min_pt_sigma_w = 0.01
    max_pt_sigma_w = 1.0
    ptsigma = (max_pt_sigma - min_pt_sigma) * x[1] + min_pt_sigma
    ptsigma_w = (max_pt_sigma_w - min_pt_sigma_w) * x[3] + min_pt_sigma_w
    ptN = stats.randint.ppf(x[4], logn_N_range[0], logn_N_range[1])
    return np.array([ptmu, ptsigma, ptmu_w, ptsigma_w, ptN])


def loglikelihood(theta, det_snr, det_width, likelihood_calc):
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
    print(real_det, config_det)
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
    likelihood_calc.convolve_p_detect()

    print("snr_thresh", snr_thresh)
    print("width_thresh", width_thresh)
    width_wide_thresh = 30e-3
    # filter the det_snr
    mask = (det_snr > snr_thresh) & (det_width > width_thresh) & (det_width < width_wide_thresh)
    det_snr = det_snr[mask]
    det_width = det_width[mask]
    likelihood_calc.calculate_pdet(det_snr,det_width)
    print(f"number of detections {len(det_snr)}")
    plot_detection_results(det_width, det_fluence, det_snr)
    print(
        "logn_N_range",
        logn_N_range,
        "logn_mu_range",
        logn_mu_range,
        "logn_std_range",
        logn_std_range,
        "logn_mu_w_range",
        logn_mu_w_range,
        "logn_std_w_range",
        logn_std_w_range,
    )
    # load the lookup table
    # xlim_lookup = np.load("xlim_second_lookup.npz",allow_pickle=1)['xlim_second']
    # mu_lookup = np.load("xlim_second_lookup.npz",allow_pickle=1)['mu']
    # std_lookup = np.load("xlim_second_lookup.npz",allow_pickle=1)['std']
    # N_lookup = np.load("xlim_second_lookup.npz",allow_pickle=1)['N']
    # mg,sg,ng = np.meshgrid(mu_lookup,std_lookup,N_lookup,indexing='ij')
    # xlim_interp = RegularGridInterpolator((mu_lookup,std_lookup,N_lookup), xlim_lookup,bounds_error=False,fill_value=None)
    nDims = 5
    # with Pool(1, loglikelihood, pt_Uniform_N, logl_args = [det_snr,xlim_interp]) as pool:
    # ln_sampler_a = dynesty.NestedSampler(pool.loglike, pool.prior_transform, nDims,
    # nlive=10000,pool=pool, queue_size=pool.njobs)
    # dill_fn = real_det.split("/")[-1]
    # dill_fn = dill_fn.split(".")[0]
    # checkpoint_fn = os.path.join(folder, f"{dill_fn}_checkpoint.h5")
    # ln_sampler_a.run_nested(checkpoint_file=checkpoint_fn)
    dill_fn = real_det.split("/")[-1]
    dill_fn = dill_fn.split(".")[:-1]
    dill_fn = ".".join(dill_fn)
    checkpoint_fn = f"{dill_fn}_logn.h5"
    print("checkpoint_fn", checkpoint_fn)
    # with Pool(1, loglikelihood, pt_Uniform_N, logl_args = [det_snr,xlim_interp]) as pool:
    #     print("starting sampling")
    #     ln_sampler_a = dynesty.NestedSampler(pool.loglike, pool.prior_transform, nDims,
    #                                         nlive=256,pool=pool, queue_size=pool.njobs)
    #     print("starting run_nested")
    #     ln_sampler_a.run_nested(checkpoint_file=checkpoint_fn)
    print("starting sampling")
    max_det = np.max(det_snr)
    max_width = np.max(det_width)
    ln_sampler_a = dynesty.NestedSampler(
        loglikelihood,
        pt_Uniform_N,
        nDims,
        logl_args=[det_snr, det_width, likelihood_calc],
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
