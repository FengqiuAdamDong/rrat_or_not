#!/usr/bin/env python3
from scipy.special import gammaln
from multiprocessing import Pool
import os
from simulate_pulse import simulate_pulses_exp
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import truncnorm
from scipy.stats import norm
import scipy
import dill

from cupyx.scipy.special import gammaln as cupy_gammaln
import statistics_basic
from statistics_basic import load_detection_fn, p_detect, p_detect_cupy
global det_error
det_error = statistics_basic.det_error
print("det_error for LN",det_error)

#################CUPY FUNCTIONS##################################
import cupy as cp
from cupyx.scipy.special import erf
def truncated_gaussian_cupy(x, mu, sigma,):
    exp_portion = cp.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * cp.sqrt(2 * cp.pi))
    constant = (1-erf(-mu/cp.sqrt(2)/sigma))/2
    return exp_portion/constant

def gaussian_cupy(x, mu, sigma):
    return cp.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * cp.sqrt(2 * cp.pi))

def gauss_second_cupy(n,mu,std,N,xlim=100,x_len=5000000):
     #xlim needs to be at least as large as 5 sigma_snrs though
    wide_enough = False
    sigma_snr = det_error
    maximum_accuracy = 1/(N-n)
    while not wide_enough:
        x_lims = [-sigma_snr*xlim,xlim]
        # x_lims = [-xlim,xlim]
        amp_arr = cp.linspace(x_lims[0],x_lims[1],x_len)
        tg_dist = truncated_gaussian_cupy(amp_arr,mu,std)
        gaussian_error = gaussian_cupy(amp_arr,0,sigma_snr)
        if (gaussian_error[-1] < maximum_accuracy)&(tg_dist[-1] < maximum_accuracy)&(gaussian_error[0] < maximum_accuracy):
            wide_enough = True
        else:
            xlim = xlim+100
    # print("second xlim",xlim)
    #convolve the two arrays
    # plt.figure()
    # plt.plot(amp_arr,LN_dist)
    # plt.plot(amp_arr,gaussian_error)
    # integral_ln = np.trapz(LN_dist,amp_arr)
    # plt.title(f"xlen {x_len} xlim {xlim} integral {integral_ln}")
    conv = cp.convolve(tg_dist,gaussian_error)*cp.diff(amp_arr)[0]
    conv_lims = [2*x_lims[0],2*x_lims[1]]
    conv_amp_array = cp.linspace(conv_lims[0],conv_lims[1],(x_len*2)-1)
    #interpolate the values for amp
    p_det = p_detect_cupy(conv_amp_array)
    likelihood = conv*(1-p_det)
    integral = cp.trapz(likelihood,conv_amp_array)
    return cp.log(integral)*(N-n)

def gauss_first_cupy(amp,mu,std,xlim=100,x_len=1000000):
    #xlim needs to be at least as large as 5 sigma_snrs though
    sigma_snr = det_error
    wide_enough = False
    while not wide_enough:
        x_lims = [-xlim,xlim]
        amp_arr = cp.linspace(x_lims[0],x_lims[1],x_len)
        tg_dist = truncated_gaussian_cupy(amp_arr,mu,std)
        gaussian_error = gaussian_cupy(amp_arr,0,sigma_snr)
        if (gaussian_error[-1] < 1e-5)&(tg_dist[-1] < 1e-5)&(xlim>(max(amp)+10)):
            wide_enough = True
        else:
            xlim = xlim+100
    # print("first xlim",xlim)
    #convolve the two arrays
    conv = cp.convolve(tg_dist,gaussian_error)*cp.diff(amp_arr)[0]
    conv_lims = [2*x_lims[0],2*x_lims[1]]
    conv_amp_array = cp.linspace(conv_lims[0],conv_lims[1],(x_len*2)-1)
    #interpolate the values for amp
    p_det = p_detect_cupy(conv_amp_array)
    likelihood_conv = conv*p_det
    likelihood = cp.interp(amp,conv_amp_array,likelihood_conv)
    return cp.sum(cp.log(likelihood))
#################CUPY END#####################

def truncated_gaussian(x, mu, sigma,):
    exp_portion = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    constant = (1-scipy.special.erf(-mu/np.sqrt(2)/sigma))/2
    return exp_portion/constant

def first_plot(amp,mu,std):
    x_len = 100000
    xlim = 100
    import pdb; pdb.set_trace()
    sigma_snr = det_error
    x_lims = [-xlim,xlim]
    amp_arr = np.linspace(x_lims[0],x_lims[1],x_len)
    gaussian_error = norm.pdf(amp_arr,0,sigma_snr)
    tg_dist = truncated_gaussian(amp_arr,mu,std)
    #convolve the two arrays
    conv = np.convolve(tg_dist,gaussian_error)*np.diff(amp_arr)[0]
    conv_lims = [-xlim*2,xlim*2]
    conv_amp_array = np.linspace(conv_lims[0],conv_lims[1],(x_len*2)-1)
    #interpolate the values for amp
    p_det = p_detect(conv_amp_array)
    likelihood_conv = conv*p_det
    likelihood = np.interp(amp,conv_amp_array,likelihood_conv)
    return likelihood, p_det, conv_amp_array, conv



def total_p(X,snr_arr=None):
    if isinstance(X,dict):
        mu = X["mu"]
        std = X["std"]
        N = X["N"]
        snr_arr = X["snr_arr"]
    else:
        mu = X[0]
        std = X[1]
        N = X[2]
    if N < len(snr_arr):
        raise Exception(" N<n")
    sigma_snr = det_error
    snr_arr = cp.array(snr_arr)
    mu = cp.array(mu)
    std = cp.array(std)
    sigma_snr = cp.array(sigma_snr)
    f = gauss_first_cupy(snr_arr, mu, std)
    if np.isnan(f):
        print("resetting f")
        f = -cp.inf
    # s = second(len(snr_arr), mu, std, N, sigma_snr=sigma_snr)
    s = gauss_second_cupy(len(snr_arr), mu, std, N)
    if np.isnan(s):
        print("resetting s")
        s = -cp.inf
    n = len(snr_arr)
    log_NCn = cupy_gammaln(N + 1) - cupy_gammaln(n + 1) - cupy_gammaln(N - n + 1)
    # print(mu,std,N)
    # print(f,s,log_NCn)
    # import pdb; pdb.set_trace()
    loglike = f + s + log_NCn
    loglike = np.array(loglike.get())
    return loglike

def negative_loglike(X, det_snr):
    x = {"k": X[0], "N": X[1], "snr_arr": det_snr}
    return -1 * total_p_gauss(x)


def likelihood_gauss(mu_arr, std_arr, N_arr, det_snr, mesh_size=20):
    # # create a mesh grid of N, mu and stds
    mat = np.zeros((mesh_size, mesh_size + 1, mesh_size + 2))
    with Pool(10) as po:

        X = []
        Y = []
        for i, mu_i in enumerate(mu_arr):
            for j, std_i in enumerate(std_arr):
                for k, N_i in enumerate(N_arr):
                    #change the mu to a different definition
                    X.append({"mu": mu_i, "std": std_i, "N": N_i, "snr_arr": det_snr})
                    Y.append([mu_i,std_i,N_i])
        Y = np.array(Y)
        # m = np.array(po.map(total_p, X))
        m = []
        for ind,v in enumerate(X):
            print(f"{ind}/{len(X)}")
            m.append(total_p(v))
        m = np.array(m)
        for i, mu_i in enumerate(mu_arr):
            for j, std_i in enumerate(std_arr):
                for k, N_i in enumerate(N_arr):
                    ind = np.sum((Y==[mu_i,std_i,N_i]),axis=1)==3
                    mat[i,j,k] = m[ind]

    return mat
