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
parser.add_argument("-i", type=str, default="fake_data.dill", help="data")
#add an argument for config file
parser.add_argument("-c", type=str, default="config.yaml", help="config file that tells you about the mu ranges, the N ranges and the k ranges")

args = parser.parse_args()
real_det = args.i
import yaml
with open(args.c, "r") as inf:
    config = yaml.safe_load(inf)
detection_curve = config["detection_curve"]
statistics_basic.load_detection_fn(detection_curve)
obs_t = config["obs_time"]
p = config["p"]
import statistics
from statistics import mean_var_to_mu_std
import statistics_exp
import statistics_gaus
##### preamble finished #####

def N_to_pfrac(x):
    total = obs_t / p
    return (1-(x / total))

def pfrac_to_N(x):
    total = obs_t / p
    return total * x

def plot_mat_exp(mat, N_arr, k_arr, fluences, dets):
    max_likelihood_exp = np.max(mat)
    mat = np.exp(mat - np.max(mat))
    fig, axes = plt.subplots(2, 2)
    posterior_N = np.trapz(mat, k_arr, axis=0)
    posterior_k = np.trapz(mat, N_arr, axis=1)
    axes[0, 0].plot(k_arr, posterior_k/np.max(posterior_k))
    max_k = k_arr[np.argmax(posterior_k)]
    axes[1, 0].pcolormesh(k_arr, N_arr, mat.T)
    axes[1, 0].set_xlabel("k")
    axes[1, 0].set_ylabel("N")
    axes[1, 1].plot(N_arr, posterior_N/np.max(posterior_N))
    max_N = N_arr[np.argmax(posterior_N)]
    axes[1, 1].set_xlabel("N")
    N_frac1 = axes[1, 1].secondary_xaxis("top", functions=(N_to_pfrac, N_to_pfrac))
    N_frac2 = axes[1, 0].secondary_yaxis("right", functions=(N_to_pfrac, N_to_pfrac))
    N_frac1.set_xlabel("Nulling Fraction")
    N_frac2.set_ylabel("Nulling Fraction")
    fig.delaxes(axes[0, 1])
    plt.tight_layout()
    plt.show()
    return max_k, max_N

def plot_mat_ln(
    mat, N_arr, mu_arr, std_arr, fluences, dets, true_mu, true_std, title="plot"
):
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

    N_frac1 = ax[2, 0].secondary_yaxis("right", functions=(N_to_pfrac, N_to_pfrac))
    N_frac2 = ax[2, 1].secondary_yaxis("right", functions=(N_to_pfrac, N_to_pfrac))
    N_frac2.set_ylabel("Nulling Fraction")
    N_frac3 = ax[2, 2].secondary_xaxis("top", functions=(N_to_pfrac, N_to_pfrac))
    N_frac3.set_xlabel("Nulling Fraction")

    fig.delaxes(ax[0, 1])
    fig.delaxes(ax[0, 2])
    fig.delaxes(ax[1, 2])
    plt.tight_layout()
    return max_mu, max_std, max_N

def plot_fit_tg(max_mu,max_std,dets):
    fit_x = np.linspace(1e-9,50,10000)
    fit_y, p_det, conv_amp_array, conv = statistics_gaus.first_plot(fit_x,max_mu,max_std)
    fit_y = fit_y/np.trapz(fit_y,fit_x)
    fig, ax = plt.subplots(1, 1)
    ax.hist(dets, bins='auto', density=True,label=f"max_mu={max_mu:.2f}, max_std={max_std:.2f}")
    ax.plot(fit_x, fit_y, label="fit")
    ax.plot(conv_amp_array, conv, label="convolution")
    ax.plot(conv_amp_array, p_det, label="p_det")
    ax.set_xlabel("SNR")
    ax.set_ylabel("Probability")
    ax.legend()
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

def plot_fit_exp(max_k,dets,sigma_det):
    fit_x = np.linspace(1e-9,50,10000)
    fit_y = statistics_exp.first_exp_plot(fit_x, max_k, sigma_det)
    fit_y = fit_y/np.trapz(fit_y,fit_x)
    fig, ax = plt.subplots(1, 1)
    ax.hist(dets, bins='auto', density=True,label=f"max_k={max_k:.2f}")
    ax.plot(fit_x, fit_y, label="fit")
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

    import pdb
    pdb.set_trace()
def calculate_bayes_factor(N_arr, mu_arr, std_arr, k_arr, mat, mat_exp):
    range_N = max(N_arr) - min(N_arr)
    range_mu = max(mu_arr) - min(mu_arr)
    range_std = max(std_arr) - min(std_arr)
    range_k = max(k_arr) - min(k_arr)

    # max_likelihood_gauss = np.max(mat_gauss)
    max_likelihood_ln = np.max(mat)
    max_likelihood_exp = np.max(mat_exp)

    # bayes_gauss = (
    #     np.log(
    #         np.trapz(
    #             np.trapz(
    #                 np.trapz(
    #                     np.exp(mat_gauss - max_likelihood_gauss), mu_arr_gauss, axis=0
    #                 ),
    #                 std_arr_gauss,
    #                 axis=0,
    #             ),
    #             N_arr_gauss,
    #             axis=0,
    #         )
    #     )
    #     + max_likelihood_gauss
    #     # - np.log(range_N * range_mu_gauss * range_std_gauss)
    # )

    bayes_ln = (
        np.log(
            np.trapz(
                np.trapz(
                    np.trapz(np.exp(mat - max_likelihood_ln), mu_arr, axis=0),
                    std_arr,
                    axis=0,
                ),
                N_arr,
                axis=0,
            )
        )
        + max_likelihood_ln
        # - np.log(range_N * range_mu * range_std)
    )
    bayes_exp = (
        np.log(
            np.trapz(
                np.trapz(np.exp(mat_exp - max_likelihood_exp), k_arr, axis=0),
                N_arr_exp,
                axis=0,
            )
        )
        + max_likelihood_exp
        # - np.log(range_N * range_k)
    )

    # print(bayes_gauss, bayes_ln, bayes_exp)
    print('LN:', bayes_ln, 'exp:', bayes_exp)
if __name__ == "__main__":
    from bayes_factor_polychord import process_detection_results, plot_detection_results,load_config
    det_fluence, det_width, det_snr, noise_std = process_detection_results(real_det)
    plot_detection_results(det_width, det_fluence, det_snr)
    #get all the things from the config file
    logn_N_range, logn_mu_range, logn_std_range, logn_mesh_size, exp_N_range, exp_k_range, exp_mesh_size, obs_t, calculate_ln, calculate_exp, det_snr, p,gauss_N_range, gauss_mu_range, gauss_std_range, gauss_mesh_size,calculate_gauss = load_config(config, det_snr)


    #this is for debugging purposes
    # calculate_matrices(det_snr)

    if calculate_gauss:
        mu_arr_gauss = np.linspace(gauss_mu_range[0], gauss_mu_range[1], gauss_mesh_size)
        std_arr_gauss = np.linspace(gauss_std_range[0], gauss_std_range[1], gauss_mesh_size + 1)
        N_arr_gauss = np.linspace(gauss_N_range[0], gauss_N_range[1], gauss_mesh_size + 2)
        mat = statistics_gaus.likelihood_gauss(mu_arr_gauss, std_arr_gauss, N_arr_gauss, det_snr, mesh_size=gauss_mesh_size)
        max_mu_gauss, max_std_gauss, max_N_gauss = plot_mat_ln(mat, N_arr_gauss, mu_arr_gauss, std_arr_gauss, det_snr, det_snr, 0, 0, title=f"Gauss num {p}")
        plot_fit_tg(max_mu_gauss, max_std_gauss, det_snr)
        plt.show()
    if calculate_ln:
        # log normal original distribution
        mu_arr = np.linspace(logn_mu_range[0], logn_mu_range[1], logn_mesh_size)
        std_arr = np.linspace(logn_std_range[0], logn_std_range[1], logn_mesh_size + 1)
        N_arr = np.linspace(logn_N_range[0], logn_N_range[1], logn_mesh_size + 2)

        mat = statistics.likelihood_lognorm(
            mu_arr, std_arr, N_arr, det_snr, mesh_size=logn_mesh_size
        )
        max_mu, max_std, max_N = plot_mat_ln(
            mat,
            N_arr,
            mu_arr,
            std_arr,
            det_snr,
            det_snr,
            0,
            0,
            title=f"Lnorm num det:{len(det_snr)}",
        )
        plot_fit_ln(max_mu, max_std, det_snr, statistics.det_error)
        plt.show()

    if calculate_exp:
        # Exponential distributions start here#####################
        k_arr = np.linspace(exp_k_range[0], exp_k_range[1], exp_mesh_size)
        N_arr_exp = np.linspace(exp_N_range[0], exp_N_range[1], exp_mesh_size + 1)
        mat_exp = statistics_exp.likelihood_exp(k_arr, N_arr_exp, det_snr)

        max_k,max_N = plot_mat_exp(mat_exp, N_arr_exp, k_arr, det_snr, det_snr)
        plot_fit_exp(max_k, det_snr, statistics.det_error)
        plt.show()

    np.savez(
        "bayes_factor_data",
        mu_arr=mu_arr,
        std_arr=std_arr,
        N_arr=N_arr,
        mat=mat,
        k_arr=k_arr,
        N_arr_exp=N_arr_exp,
        mat_exp=mat_exp,
    )
    # print("log Odds Ratio in favour of LN model", OR)
