#!/usr/bin/env python3
# paths
import sys
import numpy as np
import os
from matplotlib import pyplot as plt
from scipy import optimize as o
import dill
import warnings
import inject_stats
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
parser = argparse.ArgumentParser(description="Simulate some pulses")
#add an argument for config file
parser.add_argument("-i", default="simulated_dir", help="folder with the simulated pulses")
args = parser.parse_args()
real_det = args.i

#####preamble finished#####
cuda_device=0


def read_config(filename,det_snr):
    with open(filename, 'r') as file:
        data = yaml.safe_load(file)


    # Extract the sorted items into variables
    detection_curve = data['detection_curve']
    exp_N_range = data['exp_N_range']
    exp_k_range = data['exp_k_range']
    snr_thresh = data['snr_thresh']
    try:
        flux_cal = data['flux_cal']
    except:
        flux_cal = 1

    exp_N_range[1] = exp_N_range[1]
    if exp_N_range[0] == -1:
        #change to full range
        exp_N_range[0] = len(det_snr)+1
    return detection_curve, exp_N_range, exp_k_range, snr_thresh, flux_cal

def plot_fit_exp(max_mu,max_std,dets,sigma_det):
    fit_x = np.linspace(1e-9,50,10000)
    fit_y = statistics.first_plot(fit_x, max_mu, max_std, sigma_det)
    fig, ax = plt.subplots(1, 1)
    ax.hist(dets, bins='auto', density=True,label=f"max_mu={max_mu:.2f}, max_std={max_std:.2f}")
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


if __name__ == "__main__":
    config_det = real_det.replace(".dill",".yaml")
    with open(config_det, "r") as inf:
        config = yaml.safe_load(inf)
    detection_curve = config["detection_curve"]

    # for real_det,config_det in zip(dill_files,config_files):
    #check if png already made
    png_fp = f"{real_det}_exp_a_corner.png"
    if os.path.exists(png_fp):
        print(f"skipping {png_fp}")
        sys.exit(1)
    det_fluence, det_width, det_snr, noise_std = process_detection_results(real_det)
    print(real_det,config_det)
    detection_curve, exp_N_range, exp_k_range, snr_thresh_user, flux_cal = read_config(config_det,det_snr)
    det_snr = det_snr*flux_cal
    if snr_thresh_user > 1.6:
        snr_thresh = snr_thresh_user
    else:
        snr_thresh = 1.6
    snr_thresh = statistics_basic.load_detection_fn(detection_curve,min_snr_cutoff=snr_thresh,flux_cal=flux_cal)
    print("snr_thresh",snr_thresh)

    import statistics
    import statistics_exp
    #filter the det_snr
    det_error = statistics.det_error
    det_snr = det_snr[det_snr>snr_thresh]
    plot_detection_results(det_width, det_fluence, det_snr)
    print("exp_N_range", exp_N_range, "exp_k_range", exp_k_range)
    nDims = 2
    def pt_Uniform_N(x,max_det):
        #jeffrey's prior for ptk
        max_k = np.log(2)/(max_det/50)
        min_k = np.log(2)/max_det
        # ptk = exp_k_range[1]**x[0] / (exp_k_range[0]**(x[0]-1))
        ptk = (max_k**x[0]) / (min_k**(x[0]-1))
        ptN = (exp_N_range[1] - exp_N_range[0]) * x[1] + exp_N_range[0]
        return np.array([ptk, ptN])

    def loglikelihood(theta, det_snr):
        #print("theta",theta)
        a = 0
        lower_c = 0
        upper_c = cp.inf
        LN_k,N = (theta[0],theta[1])
        xlim = 100
        if max(det_snr) > xlim:
            xlim = max(det_snr)*2
        X = {"k": LN_k, "N": theta[1], "a":0, "lower_c":lower_c, "upper_c":upper_c}
        return statistics_exp.total_p_exp(X, snr_arr = det_snr, use_a=False,use_cutoff=True,xlim=xlim,cuda_device=cuda_device)

    def plot_fit(ln_a_sresults):
        # Plot the actual fit
        samples, weights = ln_a_sresults.samples, ln_a_sresults.importance_weights()
        mean, cov = dyfunc.mean_and_cov(samples, weights)
        max_mu = mean[0]
        max_std = mean[1]
        max_N = mean[2]
        max_a = 0
        mu, std = mean_var_to_mu_std(max_mu, max_std**2)
        fit_x = np.linspace(1e-9, 50, 10000)
        fit_y, p_det, conv_amp_array, conv = statistics.first_plot(fit_x, mu, std, det_error, a=max_a)
        fit_y = fit_y / np.trapz(fit_y, fit_x)
        fig, ax = plt.subplots(1, 1)
        ax.hist(det_snr, bins='auto', density=True, label=f"max_mu={max_mu:.2f}, max_std={max_std:.2f}")
        ax.plot(fit_x, fit_y, label="fit")
        ax.plot(conv_amp_array, conv, label="convolution")
        ax.plot(conv_amp_array, p_det, label="p_det")
        ax.set_xlabel("SNR")
        ax.set_ylabel("Probability")
        ax.legend()

    dill_fn = real_det.split("/")[-1]
    dill_fn = dill_fn.split(".")[:-1]
    dill_fn = ".".join(dill_fn)
    checkpoint_fn = f"{dill_fn}_exp.h5"
    print("checkpoint_fn",checkpoint_fn)
    # with Pool(1, loglikelihood, pt_Uniform_N, logl_args = [det_snr]) as pool:
    #     ln_sampler_a = dynesty.NestedSampler(pool.loglike, pool.prior_transform, nDims,
    #                                         nlive=256,pool=pool, queue_size=pool.njobs)
    #     ln_sampler_a.run_nested(checkpoint_file=checkpoint_fn)
    print("starting sampling")
    max_det = max(det_snr)
    ln_sampler_a = dynesty.NestedSampler(loglikelihood, pt_Uniform_N, nDims,logl_args=[det_snr],nlive=256,ptform_args=[max_det])
    print("starting run_nested")
    ln_sampler_a.run_nested(checkpoint_file=checkpoint_fn)

    ln_a_sresults = ln_sampler_a.results
    fg, ax = dyplot.cornerplot(ln_a_sresults, color='dodgerblue',labels=["k","N"], truths=np.zeros(nDims),
                            truth_color='black', show_titles=True,
                            quantiles=None, max_n_ticks=3)
    plt.savefig(f"{real_det}_exp_corner.png")
    plt.close()
