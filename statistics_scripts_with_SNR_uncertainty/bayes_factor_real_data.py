#!/usr/bin/env python3
# paths
import sys

sys.path
sys.path.append("/home/adam/Documents/rrat_or_not/injection_scripts/")

import numpy as np
import os
import statistics
from matplotlib import pyplot as plt
from scipy import optimize as o
import dill
import warnings
import inject_stats
import argparse

parser = argparse.ArgumentParser(description="Simulate some pulses")
parser.add_argument("-o", type=float, default=500, help="observed time")
parser.add_argument("-p", type=float, default=1, help="period")
parser.add_argument("-i", type=str, default="fake_data.dill", help="data")
#add an argument for config file
parser.add_argument("-c", type=str, default="config.yaml", help="config file that tells you about the mu ranges, the N ranges and the k ranges")

args = parser.parse_args()
obs_t = args.o
p = args.p
real_det = args.i
import yaml
with open(args.c, "r") as inf:
    config = yaml.safe_load(inf)
detection_curve = config["detection_curve"]
statistics.load_detection_fn(detection_curve)

import statistics_exp
import statistics_gaus
# warnings.filterwarnings("ignore")
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
    axes[0, 0].plot(k_arr, posterior_k)
    axes[1, 0].pcolormesh(k_arr, N_arr, mat.T)
    axes[1, 0].set_xlabel("k")
    axes[1, 0].set_ylabel("N")
    axes[1, 1].plot(N_arr, posterior_N)
    axes[1, 1].set_xlabel("N")
    N_frac1 = axes[1, 1].secondary_xaxis("top", functions=(N_to_pfrac, N_to_pfrac))
    N_frac2 = axes[1, 0].secondary_yaxis("right", functions=(N_to_pfrac, N_to_pfrac))
    N_frac1.set_xlabel("Nulling Fraction")
    N_frac2.set_ylabel("Nulling Fraction")
    fig.delaxes(axes[0, 1])
    plt.tight_layout()
    plt.show()


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
    ax[0, 0].plot(mu_arr, posterior_mu)
    # ax[0, 0].plot(posterior_mu)

    ax[1, 0].pcolormesh(mu_arr, std_arr, d_pos_mu_std.T)
    # ax[1, 0].imshow(d_pos_mu_std.T, aspect="auto")
    ax[1, 1].plot(std_arr, posterior_std)
    # ax[1, 1].plot(posterior_std)

    ax[2, 0].pcolormesh(mu_arr, N_arr, d_pos_mu_N.T)
    # ax[2, 0].imshow(d_pos_mu_N.T, aspect="auto")
    ax[2, 1].pcolormesh(std_arr, N_arr, d_pos_std_N.T)
    # ax[2, 1].imshow(d_pos_std_N.T, aspect="auto")
    ax[2, 2].plot(N_arr, posterior_N)
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
    plt.show()


if __name__ == "__main__":
    import sys

    odds_ratios = []
    # obs_t = 1088
    with open(real_det, "rb") as inf:
        det_class = dill.load(inf)
    det_fluence = []
    det_width = []
    det_snr = []
    noise_std = []
    for pulse_obj in det_class.sorted_pulses:
        if pulse_obj.det_amp != -1:
            # -1 means that the fluence could not be measured well
            det_fluence.append(pulse_obj.det_fluence)
            det_width.append(pulse_obj.det_std)
            det_snr.append(pulse_obj.det_snr)
            noise_std.append(pulse_obj.noise_std)
    # lets filter the det_FLUENCEs too
    det_fluence = np.array(det_fluence)
    # apply offset to det fluence

    import dill

    with open(detection_curve,"rb") as inf:
        inj_stats = dill.load(inf)
    try:
        poly_params = inj_stats.poly_snr
    except:
        #there isn't poly params, so just set to 1,0
        poly_params = [1,0]
    poly_fun = np.poly1d(poly_params)

    det_snr_altered = np.array(poly_fun(det_snr))
    det_fluence = np.array(det_fluence)
    det_width = np.array(det_width)
    det_snr = np.array(det_snr)
    det_snr = np.array(det_snr)
    noise_std = np.array(noise_std)
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
    axs[0, 1].hist(det_snr_altered, bins="auto", label="poly")
    axs[0, 1].hist(det_snr, alpha=0.5, bins="auto", label="fit")
    axs[0, 1].set_title("detected snrlitudes")
    axs[0, 1].set_xlabel("snr")
    axs[0, 1].legend()
    plt.show()
    #load the config yaml file
    logn_N_range = config["logn_N_range"]
    logn_mu_range = config["logn_mu_range"]
    logn_std_range = config["logn_std_range"]
    logn_mesh_size = config["logn_mesh_size"]

    exp_N_range = config["exp_N_range"]
    exp_k_range = config["exp_k_range"]
    exp_mesh_size = config["exp_mesh_size"]

    snr_thresh = 1.3
    print("deleting:",sum(det_snr<snr_thresh),"points")
    det_snr = det_snr[det_snr>snr_thresh]

    # plt.show()
    # det_snr = det_snr_altered


    # gaussian fit
    # mu_arr_gauss = np.linspace(-0.7, -0.6, gauss_mesh_size)
    # std_arr_gauss = np.linspace(0.1, 0.3, gauss_mesh_size + 1)
    # N_arr_gauss = np.linspace(len(det_snr), obs_t/p, gauss_mesh_size + 2)
    # N_arr_gauss = np.linspace(19000, 20100, gauss_mesh_size + 2)

    # mat_gauss = statistics_gaus.likelihood_gauss(
        # mu_arr_gauss, std_arr_gauss, N_arr_gauss, det_snr, mesh_size=gauss_mesh_size
    # )
    # plot_mat_ln(
    #     mat_gauss,
    #     N_arr_gauss,
    #     mu_arr_gauss,
    #     std_arr_gauss,
    #     det_snr,
    #     det_snr,
    #     0,
    #     0,
    #     title=f"gaussian num det={len(det_snr)}",
    # )


    # res = o.minimize(statistics.negative_loglike,[-3,0.2,obs_t/p],(det_snr),
    # method='Nelder-Mead',bounds=[(-5,5),(0.01,5),(len(det_snr),obs_t/p)])
    # mu_min = res.x[0]
    # std_min = res.x[1]
    # N_min = res.x[2]
    # print(res.x)
    # log normal original distribution
    #if logn_n_range is -1 then set to obs/p and num pulses
    if logn_N_range == -1:
        logn_N_range = [len(det_snr), obs_t / p]
    mu_arr = np.linspace(logn_mu_range[0], logn_mu_range[1], logn_mesh_size)
    std_arr = np.linspace(logn_std_range[0], logn_std_range[1], logn_mesh_size + 1)
    N_arr = np.linspace(logn_N_range[0], logn_N_range[1], logn_mesh_size + 2)
    # N_arr = np.linspace(200, obs_t/p , logn_mesh_size + 2)

    # N_arr = np.linspace(470, 1000, mesh_size + 2)
    mat = statistics.likelihood_lognorm(
        mu_arr, std_arr, N_arr, det_snr, mesh_size=logn_mesh_size
    )
    plot_mat_ln(
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


    # Exponential distributions start here#####################
    # res = o.minimize(
    # statistics_exp.negative_loglike,
    # [1, len(det_snr)],
    # (det_snr),
    # method="Nelder-Mead",
    # bounds=[(0, 100), (len(det_snr), obs_t / p)],
    # )
    # print(res.x)
    # create an array for k
    if exp_N_range == -1:
        exp_N_range = [len(det_snr), obs_t/p]
    k_arr = np.linspace(exp_k_range[0], exp_k_range[1], exp_mesh_size)
    N_arr_exp = np.linspace(exp_N_range[0], exp_N_range[1], exp_mesh_size + 1)
    mat_exp = statistics_exp.likelihood_exp(k_arr, N_arr_exp, det_snr)

    plot_mat_exp(mat_exp, N_arr_exp, k_arr, det_snr, det_snr)
    # lets calculate bayes factor
    range_N = max(N_arr) - min(N_arr)

    # Log norm range
    range_mu = max(mu_arr) - min(mu_arr)
    range_std = max(std_arr) - min(std_arr)
    # k range
    range_k = max(k_arr) - min(k_arr)
    # gauss range
    # range_mu_gauss = max(mu_arr_gauss) - min(mu_arr_gauss)
    # range_std_gauss = max(std_arr_gauss) - min(std_arr_gauss)
    # using uniform priors
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
    print('LN:',bayes_ln,'exp:',bayes_exp)
    # OR = bayes_numerator - bayes_denominator
    # if OR<0:
    #     print(f"OR less than 0 {fn}")
    #     plot_mat_ln(mat,N_arr,mu_arr,std_arr,pulse_snrs,det_snr,true_mu,true_std)
    #     plot_mat_exp(mat_exp,N_arr_exp,k_arr,pulse_snrs,det_snr)

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
