#!/usr/bin/env python3
import numpy as np
import statistics
import statistics_exp
from matplotlib import pyplot as plt


def load_data(fn):
    # load files and det snrs
    data = np.load(fn)
    # try load log norm
    try:
        snrs = data["snrs"]
        dets = data["det"]
        p = data["p"]
        obs_t = data["obs_t"]

    except:
        snrs = -1
        dets = data["det"]
        p = -1
        obs_t = -1

    return snrs, dets, p, obs_t


def plot_mat_exp(mat, N_arr, k_arr, snrs, dets):
    true_k = 0
    max_likelihood_exp = np.max(mat)
    print(max_likelihood_exp)
    mat = np.exp(mat - np.max(mat))  # *np.exp(max_likelihood_exp)
    plt.figure()
    posterior = np.trapz(mat, k_arr, axis=0)
    plt.plot(N_arr, posterior)
    plt.xlabel("N")
    plt.title(f"# of simulated pulses:{len(snrs)} # of det pulses:{len(dets)}")
    plt.figure()
    posterior = np.trapz(mat, N_arr, axis=1)
    plt.plot(k_arr, posterior)
    plt.xlabel("K")
    plt.title(f"# of simulated pulses:{len(snrs)} # of det pulses:{len(dets)}")
    plt.figure()
    # marginalise over std
    plt.pcolormesh(k_arr, N_arr, mat.T)
    plt.xlabel("k")
    plt.ylabel("N")
    plt.title(
        f"# of simulated pulses:{len(snrs)} # of det pulses:{len(dets)} true k:{true_k}"
    )
    plt.show()


def plot_mat_ln(mat, N_arr, mu_arr, std_arr, snrs, dets):
    true_mu = 0
    true_std = 0
    max_likelihood_ln = np.max(mat)
    print(max_likelihood_ln)
    mat = np.exp(mat - np.max(mat)) * np.exp(max_likelihood_ln)

    posterior = np.trapz(np.trapz(mat, mu_arr, axis=0), std_arr, axis=0)
    plt.plot(N_arr, posterior)
    plt.xlabel("N")
    plt.title(f"# of simulated pulses:{len(snrs)} # of det pulses:{len(dets)}")
    plt.show()
    posterior = np.trapz(np.trapz(mat, std_arr, axis=1), N_arr, axis=1)
    plt.plot(mu_arr, posterior)
    plt.xlabel("mu")
    plt.title(
        f"# of simulated pulses:{len(snrs)} # of det pulses:{len(dets)} true mu:{true_mu}"
    )

    plt.show()
    # marginalise over std
    d_pos = np.trapz(mat, std_arr, axis=1)
    print(d_pos.shape)
    plt.pcolormesh(mu_arr, N_arr, d_pos.T)
    plt.xlabel("mu")
    plt.ylabel("N")
    plt.title(
        f"# of simulated pulses:{len(snrs)} # of det pulses:{len(dets)} true mu:{true_mu}"
    )
    plt.show()

    d_pos = np.trapz(mat, mu_arr, axis=0)
    print(d_pos.shape)
    plt.pcolormesh(std_arr, N_arr, d_pos.T)
    plt.xlabel("std")
    plt.ylabel("N")
    plt.title(
        f"# of simulated pulses:{len(snrs)} # of det pulses:{len(dets)} true std:{true_std}"
    )
    plt.show()


if __name__ == "__main__":
    import sys

    fn = sys.argv[1]
    pulse_snrs, det_snr, p, obs_t = load_data(fn)
    N_lim = (len(det_snr), len(det_snr) * 2)
    mu_ln = np.mean(np.log10(det_snr))
    std_ln = np.std(np.log10(det_snr))
    mu_lim = (mu_ln - 1, mu_ln + 1)
    std_lim = (std_ln - 0.05, std_ln + 0.1)
    print(mu_lim, std_lim)
    # log normal original distribution
    mesh_size = 20

    mu_arr = np.linspace(mu_lim[0], mu_lim[1], mesh_size)
    std_arr = np.linspace(std_lim[0], std_lim[1], mesh_size + 1)
    N_arr = np.linspace(N_lim[0], N_lim[1], mesh_size + 2)
    mat = statistics.likelihood_lognorm(
        mu_arr, std_arr, N_arr, det_snr, mesh_size=mesh_size
    )
    plot_mat_ln(mat, N_arr, mu_arr, std_arr, pulse_snrs, det_snr)
    # create an array for k
    exp_mesh_size = 100
    k_lim = (0.001, 0.1)
    k_arr = np.linspace(k_lim[0], k_lim[1], exp_mesh_size)
    N_arr_exp = np.linspace(len(det_snr), len(pulse_snrs) * 1.2, exp_mesh_size + 1)
    mat_exp = statistics_exp.likelihood_exp(k_arr, N_arr_exp, det_snr)
    plot_mat_exp(mat_exp, N_arr_exp, k_arr, pulse_snrs, det_snr)
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
    print("log Odds Ratio in favour of LN model", bayes_numerator - bayes_denominator)
