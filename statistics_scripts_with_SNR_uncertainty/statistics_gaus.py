#!/usr/bin/env python3
from scipy.special import gammaln
from multiprocessing import Pool
import os
from simulate_pulse import simulate_pulses_exp
from statistics import n_detect
from statistics import p_detect
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import truncnorm
from scipy.stats import norm
import dill
from statistics import det_error
from statistics import inj_stats

def first_gauss(snr, mu, std, sigma_snr):
    x_len = 10000
    const = 500
    a = -mu/std
    b = (100-mu)/std


    xlim = np.abs(mu * std * const)
    if xlim<1:
        xlim=1
    x_lims = [-xlim, xlim]
    snr_true_array = np.linspace(x_lims[0], x_lims[1], x_len)
    expmodnorm = truncnorm.pdf(snr_true_array, loc=mu, scale=std,a=a,b=b)
    p_det = p_detect(snr_true_array)
    p_det_mod_norm = p_det * expmodnorm
    P_snr_true_giv_det = truncnorm.pdf(snr_true_array, 0, sigma_snr)
    conv = np.convolve(p_det_mod_norm, P_snr_true_giv_det) * np.diff(snr_true_array)[0]
    conv_lims = [-(xlim * 2), xlim * 2]
    conv_snr_array = np.linspace(conv_lims[0], conv_lims[1], (x_len * 2) - 1)
    convolve_mu_snr = np.interp(snr, conv_snr_array, conv)
    try:
        log_convolve_mu_snr = np.zeros(len(convolve_mu_snr))
        log_convolve_mu_snr[convolve_mu_snr == 0] = -np.inf
        log_convolve_mu_snr[convolve_mu_snr > 0] = np.log(
            convolve_mu_snr[convolve_mu_snr > 0]
        )
    except:
        import pdb

        pdb.set_trace()
    return np.sum(log_convolve_mu_snr)


def second_gauss(n, mu, std, N, sigma_snr):
    x_len = 10000
    const = 500
    a = -mu/std
    b = (100-mu)/std


    xlim = np.abs(mu * std * const)
    if xlim<1:
        xlim=1
    x_lims = [-xlim, xlim]

    snr = np.linspace(-xlim / 2, xlim / 2, 1000)

    snr_true_array = np.linspace(x_lims[0], x_lims[1], x_len)
    expmodnorm = truncnorm.pdf(snr_true_array, loc= mu, scale=std,a=a,b=b)
    p_det = 1 - p_detect(snr_true_array)
    p_det_mod_norm = p_det * expmodnorm

    P_snr_true_giv_det = norm.pdf(snr_true_array, 0, sigma_snr)
    conv = np.convolve(p_det_mod_norm, P_snr_true_giv_det) * np.diff(snr_true_array)[0]
    conv_lims = [-(xlim * 2), xlim * 2]
    conv_snr_array = np.linspace(conv_lims[0], conv_lims[1], (x_len * 2) - 1)
    convolve_mu_snr = np.interp(snr, conv_snr_array, conv)
    integral = np.trapz(convolve_mu_snr, snr)
    try:
        p_second_int = np.log(integral)
    except:
        import pdb

        pdb.set_trace()
    if integral > 1.01:
        print("Integral error", integral)
        p_second_int = 1
        import pdb

        pdb.set_trace()
    return p_second_int * (N - n)


def total_p_gauss(X):
    mu = X["mu"]
    std = X["std"]
    N = X["N"]
    snr_arr = X["snr_arr"]
    sigma_snr = det_error
    f = first_gauss(snr_arr, mu, std, sigma_snr=sigma_snr)
    s = second_gauss(len(snr_arr), mu, std, N, sigma_snr=sigma_snr)
    n = len(snr_arr)
    log_NCn = gammaln(N + 1) - gammaln(n + 1) - gammaln(N - n + 1)
    # print(log_NCn,f,s)
    return log_NCn + f + s


def negative_loglike(X, det_snr):
    x = {"k": X[0], "N": X[1], "snr_arr": det_snr}
    return -1 * total_p_gauss(x)


def likelihood_gauss(mu_arr, std_arr, N_arr, det_snr, mesh_size=20):
    # # create a mesh grid of N, mu and stds
    mat = np.zeros((mesh_size, mesh_size + 1, mesh_size + 2))
    with Pool(50) as po:
        for i, mu_i in enumerate(mu_arr):
            for j, std_i in enumerate(std_arr):
                X = []
                for k, N_i in enumerate(N_arr):
                    X.append({"mu": mu_i, "std": std_i, "N": N_i, "snr_arr": det_snr})
                # for ind,v in enumerate(X):
                # mat[i,j,ind] = total_p_gauss(v)
                mat[i, j, :] = po.map(total_p_gauss, X)
    return mat
