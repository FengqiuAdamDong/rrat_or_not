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
import statistics_basic
parser = argparse.ArgumentParser(description="Simulate some pulses")
parser.add_argument("-i", type=str, help="input file")
parser.add_argument("-c", type=str, help="config file")
args = parser.parse_args()
real_det = args.i
import yaml
with open(args.c, "r") as inf:
    config = yaml.safe_load(inf)
detection_curve = config["detection_curve"]
statistics_basic.load_detection_fn(detection_curve)
import statistics
from statistics import mean_var_to_mu_std

def plot_mat_ln(
    mat, N_arr, mu_arr, std_arr,title="plot"):
    # plot corner plot
    max_likelihood_ln = np.max(mat)
    mat = np.exp(mat - np.max(mat))  # *np.exp(max_likelihood_ln)
    posterior_N = np.trapz(np.trapz(mat, mu_arr, axis=0), std_arr, axis=0)
    posterior_std = np.trapz(np.trapz(mat, mu_arr, axis=0), N_arr, axis=1)
    posterior_mu = np.trapz(np.trapz(mat, std_arr, axis=1), N_arr, axis=1)
    d_pos_mu_N = np.trapz(mat, std_arr, axis=1)
    d_pos_std_N = np.trapz(mat, mu_arr, axis=0)
    d_pos_mu_std = np.trapz(mat, N_arr, axis=-1)
    fig, ax = plt.subplots(3, 3)
    fig.suptitle(title)
    ax[0, 0].plot(mu_arr, posterior_mu/np.max(posterior_mu))
    max_mu = mu_arr[np.argmax(posterior_mu)]
    # ax[0, 0].plot(posterior_mu)

    ax[1, 0].pcolormesh(mu_arr, std_arr, d_pos_mu_std.T)
    # ax[1, 0].imshow(d_pos_mu_std.T, aspect="auto")
    ax[1, 1].plot(std_arr, posterior_std/np.max(posterior_std))
    max_std = std_arr[np.argmax(posterior_std)]
    # ax[1, 1].plot(posterior_std)

    ax[2, 0].pcolormesh(mu_arr, N_arr, d_pos_mu_N.T)
    # ax[2, 0].imshow(d_pos_mu_N.T, aspect="auto")
    ax[2, 1].pcolormesh(std_arr, N_arr, d_pos_std_N.T)
    # ax[2, 1].imshow(d_pos_std_N.T, aspect="auto")
    ax[2, 2].plot(N_arr, posterior_N/np.max(posterior_N))
    max_N = N_arr[np.argmax(posterior_N)]
    ax[2, 0].set_xlabel("mu")
    ax[2, 0].set_ylabel("N")
    ax[1, 0].set_ylabel("std")
    ax[2, 1].set_xlabel("std")
    ax[2, 2].set_xlabel("N")


    fig.delaxes(ax[0, 1])
    fig.delaxes(ax[0, 2])
    fig.delaxes(ax[1, 2])
    plt.tight_layout()
    return max_mu, max_std, max_N

def plot_fit_ln(max_mu,max_std,dets,sigma_det):
    mu, std = mean_var_to_mu_std(max_mu, max_std**2)
    fit_x = np.linspace(1e-9,50,10000)
    fit_y, p_det, conv_amp_array, conv = statistics.first_plot(fit_x, mu, std, sigma_det)
    fit_y = fit_y/np.trapz(fit_y,fit_x)
    fig, ax = plt.subplots(1, 1)
    ax.hist(dets, bins='auto', density=True,label=f"max_mu={max_mu:.2f}, max_std={max_std:.2f}")
    ax.plot(fit_x, fit_y, label="fit")
    ax.plot(conv_amp_array, conv, label="convolution")
    ax.plot(conv_amp_array, p_det, label="p_det")
    ax.set_xlabel("SNR")
    ax.set_ylabel("Probability")
    ax.legend()

def calculate_matrices(det_snr):
    xlim = np.linspace(45, 80, 5)
    xlen = np.linspace(200000, 1000000, 10, dtype=int)

    first_mat = np.zeros((len(xlim), len(xlen)))
    second_mat = np.zeros((len(xlim), len(xlen)))

    for xlim_idx in range(len(xlim)):
        for xlen_idx in range(len(xlen)):
            first_mat[xlim_idx, xlen_idx] = statistics.first_cupy(cp.array(det_snr), -3, 1.64, xlim=xlim[xlim_idx], x_len=xlen[xlen_idx])
            second_mat[xlim_idx, xlen_idx] = statistics.second_integral_cupy(len(det_snr), -3, 1.64, 1e5, xlim=xlim[xlim_idx], x_len=xlen[xlen_idx])
            print(xlim_idx, xlen_idx)

    plt.figure()
    plt.imshow(first_mat, extent=[10000, 50000, 5, 40], aspect='auto')
    plt.title("first")
    plt.colorbar()

    plt.figure()
    plt.imshow(second_mat, extent=[10000, 50000, 5, 40], aspect='auto')
    plt.title("second")
    plt.colorbar()

    plt.show()
def load_config(config, det_snr):
    logn_N_range = config["logn_N_range"]
    logn_mu_range = config["logn_mu_range"]
    logn_std_range = config["logn_std_range"]
    if logn_N_range[0] == -1:
        logn_N_range[0] = len(det_snr)+1
    try:
        logn_mesh_size = config["logn_mesh_size"]
    except:
        logn_mesh_size = 5
    try:
        obs_t = config["obs_time"]
        p = config["p"]
    except:
        obs_t = 1
        p = 1
    snr_thresh = config["snr_thresh"]
    print("deleting:", sum(det_snr < snr_thresh), "points")
    det_snr = det_snr[det_snr > snr_thresh]

    if logn_N_range == -1:
        logn_N_range = [len(det_snr), obs_t / p]

    return logn_N_range, logn_mu_range, logn_std_range, logn_mesh_size, obs_t, snr_thresh, det_snr, p

if __name__ == "__main__":
    from bayes_factor_polychord import process_detection_results, plot_detection_results
    det_fluence, det_width, det_snr, noise_std = process_detection_results(real_det)
    plot_detection_results(det_width, det_fluence, det_snr)
    #get all the things from the config file
    logn_N_range, logn_mu_range, logn_std_range, logn_mesh_size, obs_t, snr_thresh, det_snr, p = load_config(config, det_snr)

    # log normal original distribution
    mu_arr = np.linspace(logn_mu_range[0], logn_mu_range[1], logn_mesh_size)
    std_arr = np.linspace(logn_std_range[0], logn_std_range[1], logn_mesh_size + 1)
    N_arr = np.linspace(logn_N_range[0], logn_N_range[1], logn_mesh_size +2)

    mat = statistics.likelihood_lognorm(
        mu_arr, std_arr, N_arr, det_snr, mesh_size=logn_mesh_size
    )
    max_mu, max_std, max_N = plot_mat_ln(
        mat,
        N_arr,
        mu_arr,
        std_arr,
        title=f"Lnorm num det:{len(det_snr)}",
    )
    fn = real_det.split("/")[-1].split(".")[0]
    savefn = f"bayes_factor_{fn}.npz"
    print(savefn)
    plt.savefig(f"bayes_factor_{fn}.png")
    np.savez(savefn, mu_arr=mu_arr, std_arr=std_arr, N_arr=N_arr, mat=mat)
    plt.close()
