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
parser = argparse.ArgumentParser(description="Simulate some pulses")
parser.add_argument("-i", type=str, default="fake_data.dill", help="data")
#add an argument for config file
parser.add_argument("-c", type=str, default="config.yaml", help="config file that tells you about the mu ranges, the N ranges and the k ranges")
parser.add_argument("-resume", default=False, help="Tries to see if there was an old run and whether you should keep running", action="store_true")

args = parser.parse_args()
real_det = args.i
resume = args.resume
import yaml
with open(args.c, "r") as inf:
    config = yaml.safe_load(inf)
detection_curve = config["detection_curve"]
statistics_basic.load_detection_fn(detection_curve)

import statistics
from statistics import mean_var_to_mu_std
import statistics_exp
det_error = statistics.det_error
# import statistics_gaus
#####preamble finished#####






# warnings.filterwarnings("ignore")
def N_to_pfrac(x):
    total = obs_t / p
    return (1-(x / total))

def pfrac_to_N(x):
    total = obs_t / p
    return total * x

def plot_fit_ln(max_mu,max_std,dets,sigma_det):
    fit_x = np.linspace(1e-9,50,10000)
    fit_y = statistics.first_plot(fit_x, max_mu, max_std, sigma_det)
    fig, ax = plt.subplots(1, 1)
    ax.hist(dets, bins='auto', density=True,label=f"max_mu={max_mu:.2f}, max_std={max_std:.2f}")
    ax.plot(fit_x, fit_y, label="fit")
    ax.set_xlabel("SNR")
    ax.set_ylabel("Probability")
    ax.legend()
    plt.show()

def plot_fit_exp(max_k,dets,sigma_det):
    fit_x = np.linspace(1e-9,50,10000)
    fit_y = statistics_exp.first_exp_plot(fit_x, max_k, sigma_det)
    fig, ax = plt.subplots(1, 1)
    ax.hist(dets, bins='auto', density=True,label=f"max_k={max_k:.2f}")
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

def load_config(config, det_snr):
    logn_N_range = config["logn_N_range"]
    logn_mu_range = config["logn_mu_range"]
    logn_std_range = config["logn_std_range"]
    logn_mesh_size = config["logn_mesh_size"]

    gauss_N_range = config["gauss_N_range"]
    gauss_mu_range = config["gauss_mu_range"]
    gauss_std_range = config["gauss_std_range"]
    gauss_mesh_size = config["gauss_mesh_size"]

    exp_N_range = config["exp_N_range"]
    exp_k_range = config["exp_k_range"]
    exp_mesh_size = config["exp_mesh_size"]
    obs_t = config["obs_time"]

    calculate_ln = config["calculate_ln"]
    calculate_exp = config["calculate_exp"]
    calculate_gauss = config["calculate_gauss"]
    snr_thresh = config["snr_thresh"]
    print("deleting:", sum(det_snr < snr_thresh), "points")
    det_snr = det_snr[det_snr > snr_thresh]
    p = config["p"]

    if logn_N_range == -1:
        logn_N_range = [len(det_snr), obs_t / p]
    if exp_N_range == -1:
        exp_N_range = [len(det_snr), obs_t / p]
    if gauss_N_range == -1:
        gauss_N_range = [len(det_snr), obs_t / p]

    return logn_N_range, logn_mu_range, logn_std_range, logn_mesh_size, exp_N_range, exp_k_range, exp_mesh_size, obs_t, calculate_ln, calculate_exp, det_snr, p, gauss_N_range, gauss_mu_range, gauss_std_range, gauss_mesh_size, calculate_gauss

if __name__ == "__main__":
    det_fluence, det_width, det_snr, noise_std = process_detection_results(real_det)
    plot_detection_results(det_width, det_fluence, det_snr)
    logn_N_range, logn_mu_range, logn_std_range, logn_mesh_size, exp_N_range, exp_k_range, exp_mesh_size, obs_t, calculate_ln, calculate_exp, det_snr, p, gauss_N_range, gauss_mu_range, gauss_std_range, gauss_mesh_size, calculate_gauss = load_config(config, det_snr)

    print("logn_N_range", logn_N_range, "logn_mu_range", logn_mu_range, "logn_std_range", logn_std_range)
    print("exp_N_range", exp_N_range, "exp_k_range", exp_k_range)


    # if calculate_ln:
    if False:
        nDims = 3
        def pt_Uniform_N(x):
            ptmu = (logn_mu_range[1] - logn_mu_range[0]) * x[0] + logn_mu_range[0]
            ptsigma = (logn_std_range[1] - logn_std_range[0]) * x[1] + logn_std_range[0]
            ptN = (logn_N_range[1] - logn_N_range[0]) * x[2] + logn_N_range[0]
            return np.array([ptmu, ptsigma, ptN])

        def loglikelihood(theta, det_snr):
            theta[0],theta[1] = mean_var_to_mu_std(theta[0], theta[1]**2)
            return statistics.total_p(theta, det_snr)

        with Pool(20, loglikelihood, pt_Uniform_N, logl_args = [det_snr]) as pool:
            ln_sampler = dynesty.NestedSampler(pool.loglike, pool.prior_transform, nDims,
                                               nlive=256,pool=pool, queue_size=10)
            ln_sampler.run_nested(checkpoint_file=f"{real_det}_logn_checkpoint.h5")
        ln_sresults = ln_sampler.results


        # rfig, raxes = dyplot.runplot(sresults)
        # Plot the 2-D marginalized posteriors.
        cfig, caxes = dyplot.cornerplot(ln_sresults)
        plt.title("logn")

    if calculate_ln:
        #see if a checkpoint file exists
        xlim_lookup = np.load("xlim_second_lookup.npz",allow_pickle=1)['xlim_second']
        mu_lookup = np.load("xlim_second_lookup.npz",allow_pickle=1)['mu']
        std_lookup = np.load("xlim_second_lookup.npz",allow_pickle=1)['std']
        N_lookup = np.load("xlim_second_lookup.npz",allow_pickle=1)['N']
        mg,sg,ng = np.meshgrid(mu_lookup,std_lookup,N_lookup,indexing='ij')
        xlim_interp = RegularGridInterpolator((mu_lookup,std_lookup,N_lookup), xlim_lookup,bounds_error=False,fill_value=None)
        nDims = 4

        def pt_Uniform_N(x):
            ptmu = (logn_mu_range[1] - logn_mu_range[0]) * x[0] + logn_mu_range[0]
            ptsigma = (logn_std_range[1] - logn_std_range[0]) * x[1] + logn_std_range[0]
            ptN = (logn_N_range[1] - logn_N_range[0]) * x[2] + logn_N_range[0]
            a_range = [0,np.mean(det_snr)]
            # a_range = [0,1]
            pta = (a_range[1] - a_range[0]) * x[3] + a_range[0]
            return np.array([ptmu, ptsigma, ptN, pta])

        def loglikelihood(theta, det_snr, xlim_interp):
            #convert to strict upper limit of the lognorm
            #convert to the standard mu and sigma of a lognorm
            theta = np.append(theta,0)
            theta = np.append(theta,theta[0]*50)
            theta[0],theta[1] = mean_var_to_mu_std(theta[0], theta[1]**2)
            xlim = xlim_interp([[theta[0],theta[1],theta[2]]])[0]
            return statistics.total_p(theta, snr_arr = det_snr, use_a=True,use_cutoff=True,xlim=xlim)

        def plot_fit(ln_a_sresults):
            # Plot the actual fit
            samples, weights = ln_a_sresults.samples, ln_a_sresults.importance_weights()
            mean, cov = dyfunc.mean_and_cov(samples, weights)
            max_mu = mean[0]
            max_std = mean[1]
            max_N = mean[2]
            max_a = mean[3]
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

        if resume:
            if os.path.isfile(f"{real_det}_logn_a_checkpoint.h5"):
                print("resuming previous calculation")
                filepath = f"{real_det}_logn_a_checkpoint.h5"
                with Pool(1, loglikelihood, pt_Uniform_N, logl_args = [det_snr,xlim_interp]) as pool:
                    ln_sampler_a = dynesty.NestedSampler.restore(filepath,pool=pool)
                    ln_sampler_a.run_nested(resume = True,checkpoint_file=f"{real_det}_logn_a_checkpoint.h5")
                res2 = ln_sampler_a.results
                samples, weights = res2.samples, res2.importance_weights()
                mean, cov = dyfunc.mean_and_cov(samples, weights)
                # plot extended run (res2; right)
                fg, ax = dyplot.cornerplot(res2, color='dodgerblue', truths=np.zeros(nDims),
                                        truth_color='black', show_titles=True,
                                        quantiles=None, max_n_ticks=3)
                plot_fit(res2)
                plt.show()

        else:
            with Pool(1, loglikelihood, pt_Uniform_N, logl_args = [det_snr,xlim_interp]) as pool:
                ln_sampler_a = dynesty.NestedSampler(pool.loglike, pool.prior_transform, nDims,
                                                    nlive=256,pool=pool, queue_size=pool.njobs)
                ln_sampler_a.run_nested(checkpoint_file=f"{real_det}_logn_a_checkpoint.h5")
            # ln_sampler_a = dynesty.NestedSampler(loglikelihood, pt_Uniform_N, nDims,
                                    # logl_args=[det_snr,xlim_interp], nlive=256)
            # ln_sampler_a.run_nested(checkpoint_file=f"{real_det}_logn_a_checkpoint.h5")
            # ln_sampler_a = dynesty.DynamicNestedSampler(loglikelihood, pt_Uniform_N, nDims,
                                    # logl_args=[det_snr,xlim_interp])
            # ln_sampler_a.run_nested(checkpoint_file=f"{real_det}_logn_a_checkpoint.h5",nlive_init=20,nlive_batch=20,dlogz_init=200)

            ln_a_sresults = ln_sampler_a.results
            fg, ax = dyplot.cornerplot(ln_a_sresults, color='dodgerblue', truths=np.zeros(nDims),
                                    truth_color='black', show_titles=True,
                                    quantiles=None, max_n_ticks=3)
            plot_fit(ln_a_sresults)

    plt.show()
    import pdb; pdb.set_trace()

    if calculate_exp:
        nDims = 2

        def pt_Uniform_K(x):
            ptK = (exp_k_range[1] - exp_k_range[0]) * x[0] + exp_k_range[0]
            ptN = (exp_N_range[1] - exp_N_range[0]) * x[1] + exp_N_range[0]
            return np.array([ptK, ptN])

        def exp_loglikelihood(theta, det_snr):
            return statistics_exp.total_p_exp(theta, det_snr)

        exp_sampler = dynesty.NestedSampler(exp_loglikelihood, pt_Uniform_K, ndim=nDims,
                                            logl_args=[det_snr], nlive=10000)
        exp_sampler.run_nested(checkpoint_file=f"{real_det}_exp_checkpoint.h5")
        exp_sresults = exp_sampler.results
        cfig, caxes = dyplot.cornerplot(exp_sresults)
        plt.title("exp")

