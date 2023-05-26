#!/usr/bin/env python3
# paths
import sys

sys.path
sys.path.append("/home/adam/Documents/rrat_or_not/injection_scripts/")

import numpy as np
import os
import statistics
import statistics_exp
import statistics_gaus
from matplotlib import pyplot as plt
from scipy import optimize as o
import dill
import warnings
import inject_stats
import argparse

parser = argparse.ArgumentParser(description="Simulate some pulses")
parser.add_argument("-o", type=float, default=500, help="mean")
parser.add_argument("-p", type=float, default=1, help="standard deviation")
parser.add_argument("-i", type=str, default="fake_data.dill", help="data")

args = parser.parse_args()
obs_t = args.o
p = args.p
real_det = args.i


# warnings.filterwarnings("ignore")
def N_to_pfrac(x):
    total = obs_t / p
    return x / total


def pfrac_to_N(x):
    total = obs_t / p
    return total * x


def plot_mat_exp(mat, N_arr, k_arr, fluences, dets):
    true_k = 0
    max_likelihood_exp = np.max(mat)
    mat = np.exp(mat - np.max(mat))  # *np.exp(max_likelihood_exp)
    fig1, ax1 = plt.subplots()
    posterior = np.trapz(mat, k_arr, axis=0)
    ax1.plot(N_arr, posterior)
    secax1 = ax1.secondary_xaxis("top", functions=(N_to_pfrac, N_to_pfrac))
    secax1.set_xlabel("Pulsing Fraction")
    ax1.set_xlabel("N")
    ax1.set_title(f"# of det pulses:{len(dets)}")

    plt.figure()
    posterior = np.trapz(mat, N_arr, axis=1)
    plt.plot(k_arr, posterior)
    plt.xlabel("K")
    plt.title(f"# of det pulses:{len(dets)}")

    fig3, ax3 = plt.subplots()
    # marginalise over std
    plt.pcolormesh(k_arr, N_arr, mat.T)
    secax3 = ax3.secondary_yaxis("right", functions=(N_to_pfrac, N_to_pfrac))
    secax3.set_ylabel("Pulsing Fraction")
    ax3.set_xlabel("k")
    ax3.set_ylabel("N")
    ax3.set_title(f"# of det pulses:{len(dets)}")
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
    ax[1, 0].pcolormesh(mu_arr, std_arr, d_pos_mu_std.T)
    ax[1, 1].plot(std_arr, posterior_std)
    ax[2, 0].pcolormesh(mu_arr, N_arr, d_pos_mu_N.T)
    ax[2, 1].pcolormesh(std_arr, N_arr, d_pos_std_N.T)
    ax[2, 2].plot(N_arr, posterior_N)
    ax[2, 0].set_xlabel("mu")
    ax[2, 0].set_ylabel("N")
    ax[1, 0].set_ylabel("std")
    ax[2, 1].set_xlabel("std")
    ax[2, 2].set_xlabel("N")

    N_frac1 = ax[2, 0].secondary_yaxis("right", functions=(N_to_pfrac, N_to_pfrac))
    N_frac2 = ax[2, 1].secondary_yaxis("right", functions=(N_to_pfrac, N_to_pfrac))
    N_frac2.set_ylabel("Pulsing Fraction")
    N_frac3 = ax[2, 2].secondary_xaxis("top", functions=(N_to_pfrac, N_to_pfrac))
    N_frac3.set_xlabel("Pulsing Fraction")

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
    det_amp = []
    det_amp_fluence = []
    for pulse_obj in det_class.sorted_pulses:
        if pulse_obj.det_fluence != -1:
            # -1 means that the fluence could not be measured well
            det_fluence.append(pulse_obj.det_fluence)
            det_width.append(pulse_obj.det_std)
            det_amp.append(pulse_obj.det_amp)
            det_amp_fluence.append(pulse_obj.fluence_amp)
    # lets filter the det_FLUENCEs too
    det_fluence = np.array(det_fluence) * 1e3
    # apply offset to det fluence
    poly_params = np.load("det_fun_params.npz")["poly"]
    poly_fun = np.poly1d(poly_params)
    det_fluence = poly_fun(det_fluence)
    det_width = np.array(det_width)
    print(np.mean(det_width))
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].set_title("Widths")
    axs[0, 0].hist(det_width, bins=50)

    axs[1, 0].set_title("Histogram of detected pulses")
    axs[1, 0].hist(det_fluence, bins="auto", density=True)
    axs[1, 0].set_xlabel("FLUENCE")
    axs[1, 1].hist(det_amp_fluence, bins="auto", label="fluence")
    axs[1, 1].hist(det_amp, alpha=0.5, bins="auto", label="fit")
    axs[1, 1].set_title("detected amplitudes")
    axs[1, 1].set_xlabel("amp")
    axs[1, 1].legend()
    plt.show()
    det_fluence = np.array(det_fluence)

    gauss_mesh_size = 10
    mesh_size = 20
    exp_mesh_size = 150

    # gaussian fit
    mu_arr_gauss = np.linspace(0.5, 3, gauss_mesh_size)
    std_arr_gauss = np.linspace(0.1, 1.5, gauss_mesh_size + 1)
    N_arr_gauss = np.linspace(52, 2000, gauss_mesh_size + 2)
    mat_gauss = statistics_gaus.likelihood_gauss(
        mu_arr_gauss, std_arr_gauss, N_arr_gauss, det_fluence, mesh_size=gauss_mesh_size
    )
    plot_mat_ln(
        mat_gauss,
        N_arr_gauss,
        mu_arr_gauss,
        std_arr_gauss,
        det_fluence,
        det_fluence,
        0,
        0,
        title=f"gaussian num det={len(det_fluence)}",
    )

    # res = o.minimize(statistics.negative_loglike,[0.5,0.2,len(det_fluence)],(det_fluence),
    #             method='Nelder-Mead',bounds=[(0.001,2),(0.1,2),(len(det_fluence),100000)])
    # mu_min = res.x[0]
    # std_min = res.x[1]
    # N_min = res.x[2]
    # fluence_s = np.linspace(1e-2,25,1000)
    # y = statistics.lognorm_dist(fluence_s,mu_min,std_min)*statistics.p_detect(fluence_s)
    # y_scale = np.trapz(y,fluence_s)
    # y = y/y_scale

    # plt.plot(fluence_s,y)
    # plt.show()

    # print(mu_min,std_min,N_min)

    # log normal original distribution
    mu_arr = np.linspace(0.001, 0.5, mesh_size)
    std_arr = np.linspace(0.2, 0.6, mesh_size + 1)
    N_arr = np.linspace(60, 2000, mesh_size + 2)

    mat = statistics.likelihood_lognorm(
        mu_arr, std_arr, N_arr, det_fluence, mesh_size=mesh_size
    )
    plot_mat_ln(
        mat,
        N_arr,
        mu_arr,
        std_arr,
        det_fluence,
        det_fluence,
        0,
        0,
        title=f"Lnorm num det:{len(det_fluence)}",
    )

    # Exponential distributions start here#####################
    # find the minimum for the exp
    res = o.minimize(
        statistics_exp.negative_loglike,
        [1, len(det_fluence)],
        (det_fluence),
        method="Nelder-Mead",
        bounds=[(0, 10), (len(det_fluence), 4 * obs_t / p)],
    )
    print(res.x)
    plt.title("Histogram of detected pulses")
    n, bins, patches = plt.hist(det_fluence, bins="auto", density=True)
    plt.xlabel("FLUENCE")
    plt.ylabel("count")

    # create an array for k
    min_k = res.x[0]
    min_N = res.x[1]
    y = np.exp(statistics_exp.log_fluence_distribution_exp(fluence_s, min_k))
    scale = n[0] / np.exp(
        statistics_exp.log_fluence_distribution_exp(bins[0] + np.diff(bins)[0], min_k)
    )
    y = y * scale
    plt.plot(fluence_s, y)
    plt.show()

    k_arr = np.linspace(1.5, 2.5, exp_mesh_size)
    N_arr_exp = np.linspace(len(det_fluence), obs_t / p, exp_mesh_size * 3)
    mat_exp = statistics_exp.likelihood_exp(k_arr, N_arr_exp, det_fluence)
    plot_mat_exp(mat_exp, N_arr_exp, k_arr, det_fluence, det_fluence)
    # lets calculate bayes factor
    range_N = max(N_arr) - min(N_arr)
    range_mu = max(mu_arr) - min(mu_arr)
    range_std = max(std_arr) - min(std_arr)
    # using uniform priors
    max_likelihood_ln = np.max(mat)
    max_likelihood_exp = np.max(mat_exp)
    bayes_numerator = (
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
        - np.log(1 / (range_N * range_mu * range_std))
    )
    bayes_denominator = (
        np.log(
            np.trapz(
                np.trapz(np.exp(mat_exp - max_likelihood_exp), k_arr, axis=0),
                N_arr_exp,
                axis=0,
            )
        )
        + max_likelihood_exp
        - np.log(1 / (range_N * range_mu))
    )
    OR = bayes_numerator - bayes_denominator
    # if OR<0:
    #     print(f"OR less than 0 {fn}")
    #     plot_mat_ln(mat,N_arr,mu_arr,std_arr,pulse_fluences,det_fluence,true_mu,true_std)
    #     plot_mat_exp(mat_exp,N_arr_exp,k_arr,pulse_fluences,det_fluence)

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
    print("log Odds Ratio in favour of LN model", OR)
