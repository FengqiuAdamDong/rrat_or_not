#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from math import comb
from scipy.special import gammaln
from multiprocessing import Pool
import os
import scipy.optimize as opt
import dill
from scipy.integrate import quad

with open("inj_stats_fitted.dill", "rb") as inf:
    inj_stats = dill.load(inf)
# popt = inj_stats.fit_logistic_amp
det_error = inj_stats.detect_error_amp

snr_arr = np.linspace(-2, 2, 1000)
print("det error", det_error)


def lognorm_dist(x, mu, sigma):
    pdf = np.zeros(x.shape)
    pdf[x > 0] = np.exp(-((np.log(x[x > 0]) - mu) ** 2) / (2 * sigma**2)) / (
        x[x > 0] * sigma * np.sqrt(2 * np.pi)
    )
    return pdf


def logistic(x, k, x0):
    L = 1
    snr = x
    detection_fn = np.zeros(len(snr))
    snr_limit = 1
    detection_fn[(snr > -snr_limit) & (snr < snr_limit)] = L / (
        1 + np.exp(-k * (snr[(snr > -snr_limit) & (snr < snr_limit)] - x0))
    )
    detection_fn[snr >= snr_limit] = 1
    detection_fn[snr <= -snr_limit] = 0
    return detection_fn

def p_detect(snr):
    return inj_stats.predict_poly(snr,inj_stats.detected_bin_midpoints,inj_stats.detected_det_frac)

detfn = p_detect(snr_arr)
plt.plot(snr_arr, detfn)


def n_detect(snr_emit):
    # snr emit is the snr that the emitted pulse has
    p = p_detect(snr_emit)
    # simulate random numbers between 0 and 1
    rands = np.random.rand(len(p))
    # probability the random number is less than p gives you an idea of what will be detected
    detected = snr_emit[rands < p]
    return detected

def first(amp,mu,std, sigma_snr=0.4):
    x_len = 10000
    xlim=2
    x_lims = [-xlim,xlim]
    amp_arr = np.linspace(x_lims[0],x_lims[1],x_len)
    gaussian_error = norm.pdf(amp_arr,0,sigma_snr)
    LN_dist = lognorm_dist(amp_arr,mu,std)
    #convolve the two arrays
    conv = np.convolve(LN_dist,gaussian_error)*np.diff(amp_arr)[0]
    conv_lims = [-xlim*2,xlim*2]
    conv_amp_array = np.linspace(conv_lims[0],conv_lims[1],(x_len*2)-1)
    #interpolate the values for amp
    p_det = inj_stats.predict_poly(conv_amp_array,inj_stats.detected_bin_midpoints,inj_stats.detected_det_frac)

    likelihood_conv = conv*p_det
    likelihood = np.interp(amp,conv_amp_array,likelihood_conv)
    # plt.close()
    # plt.hist(amp,density=True)
    # plt.plot(conv_amp_array,likelihood_conv/np.trapz(likelihood_conv,conv_amp_array))
    # plt.plot(amp_arr,LN_dist,label="ln_dist")
    # plt.legend()
    # plt.show()
    return np.sum(np.log(likelihood))

def second(n, mu, std, N, sigma_snr):

    x_len = 10000
    xlim=2
    x_lims = [-xlim,xlim]
    amp = np.linspace(-xlim / 2, xlim / 2, 10000)

    amp_arr = np.linspace(x_lims[0],x_lims[1],x_len)
    gaussian_error = norm.pdf(amp_arr,0,sigma_snr)
    p_not_det = 1-inj_stats.predict_poly(amp,inj_stats.detected_bin_midpoints,inj_stats.detected_det_frac)
    # p_det_giv_param = p_detect(amp_arr)*norm.pdf(amp_arr,mu,std)
    LN_dist = lognorm_dist(amp_arr,mu,std)
    # plt.figure()
    # plt.plot(amp_arr,LN_dist)
    # plt.show()
    #convolve the two arrays
    conv = np.convolve(LN_dist,gaussian_error)*np.diff(amp_arr)[0]
    conv_lims = [-xlim*2,xlim*2]
    conv_amp_array = np.linspace(conv_lims[0],conv_lims[1],(x_len*2)-1)
    #interpolate the values for amp
    p_det = inj_stats.predict_poly(conv_amp_array,inj_stats.detected_bin_midpoints,inj_stats.detected_det_frac)
    likelihood_conv = conv*(1-p_det)
    likelihood = np.interp(amp,conv_amp_array,likelihood_conv)

    integral = np.trapz(likelihood,amp)
    try:
        p_second_int = np.log(integral)
    except:
        import pdb
        pdb.set_trace()
    if integral > 1:
        print("Integral error", integral)
        # p_second_int = 1

    return p_second_int * (N - n)


def total_p(X):
    mu = X["mu"]
    std = X["std"]
    N = X["N"]
    snr_arr = X["snr_arr"]
    if N < len(snr_arr):
        raise Exception(" N<n")
    sigma_snr = det_error
    f = first(snr_arr, mu, std, sigma_snr=sigma_snr)
    if np.isnan(f):
        print("resetting f")
        f = -np.inf
    s = second(len(snr_arr), mu, std, N, sigma_snr=sigma_snr)
    if np.isnan(s):
        print("resetting s")
        s = -np.inf
    n = len(snr_arr)
    log_NCn = gammaln(N + 1) - gammaln(n + 1) - gammaln(N - n + 1)
    # print(mu,std,N)
    # print(f,s,log_NCn)
    # import pdb; pdb.set_trace()
    return log_NCn + f + s


def negative_loglike(X, det_snr):
    x = {"mu": X[0], "std": X[1], "N": X[2], "snr_arr": det_snr}
    return -1 * total_p(x)


def likelihood_lognorm(mu_arr, std_arr, N_arr, det_snr, mesh_size=20):
    # # create a mesh grid of N, mu and stds
    mat = np.zeros((mesh_size, mesh_size + 1, mesh_size + 2))
    with Pool(70) as po:

        X = []
        Y = []
        for i, mu_i in enumerate(mu_arr):
            for j, std_i in enumerate(std_arr):
                for k, N_i in enumerate(N_arr):
                    X.append({"mu": mu_i, "std": std_i, "N": N_i, "snr_arr": det_snr})
                    Y.append([mu_i,std_i,N_i])
        Y = np.array(Y)
        m = np.array(po.map(total_p, X))
        # m = []
        # for ind,v in enumerate(X):
            # m.append(total_p(v))
        # m = np.array(m)

        for i, mu_i in enumerate(mu_arr):
            for j, std_i in enumerate(std_arr):
                for k, N_i in enumerate(N_arr):
                    ind = np.sum((Y==[mu_i,std_i,N_i]),axis=1)==3
                    mat[i,j,k] = m[ind]

    return mat


