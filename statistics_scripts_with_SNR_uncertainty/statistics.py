#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from math import comb
from scipy.special import gammaln
from multiprocessing import Pool
import os
import dill
import scipy
from cupyx.scipy.special import gammaln as cupy_gammaln
from cupyx.scipy.special import erf as cupy_erf
import statistics_basic
from statistics_basic import load_detection_fn, p_detect_cupy, p_detect_cpu
global det_error
det_error = statistics_basic.det_error
print("det_error for LN",det_error)
import time
###############################CUPY FUNCTIONS##################################
import cupy as cp

def lognorm_dist_cupy(x, mu, sigma, lower_c=0, upper_c=cp.inf):
    #lower and upper cutoff parameters added
    pdf = cp.zeros(x.shape)
    mask = (x > lower_c) & (x < upper_c)
    pdf[mask] = cp.exp(-((cp.log(x[mask]) - mu) ** 2) / (2 * sigma**2)) / (
        (x[mask]) * sigma * cp.sqrt(2 * cp.pi)
    )
    def argument(c,mu,sigma):
        if c==0:
            return -cp.inf
        return (cp.log(c)-mu)/(sigma*cp.sqrt(2))

    pdf = 2*pdf / (cupy_erf(argument(upper_c,mu,sigma))-cupy_erf(argument(lower_c,mu,sigma)))
    return pdf

def gaussian_cupy(x, mu, sigma):
    return cp.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * cp.sqrt(2 * cp.pi))

def second_cupy(n,mu,std,N,xlim=10,x_len=1000,a=0,lower_c=0,upper_c=cp.inf):
     #xlim needs to be at least as large as 5 sigma_snrs though
    #xlim needs to be at least as large as 5 sigma_snrs though
    sigma_lim = 6
    upper = mu + (sigma_lim * std)
    lower = mu - (sigma_lim * std)
    amp_ln = cp.linspace(lower, upper, 1001)
    amp = cp.exp(amp_ln)
    sigma_amp = det_error
    # amp is the detected amps
    # width is the detected widths
    # make an array of lower and upper limits for the true_log amp array
    sigma_lim = 5
    true_lower = amp - sigma_lim * sigma_amp
    true_lower[true_lower < 0] = cp.exp(-20)
    true_upper = amp + sigma_lim * sigma_amp
    # generate a mesh of amps
    # true_amp_mesh = cp.zeros((len(amp), x_len))
    true_amp_mesh = cp.linspace(cp.log(true_lower),cp.log(true_upper),x_len).T

    amp = amp[:, cp.newaxis]
    gaussian_error_amp = gaussian_cupy(amp, cp.exp(true_amp_mesh), sigma_amp)
    lognorm_amp_dist = gaussian_cupy(true_amp_mesh, mu, std)
    mult_amp = gaussian_error_amp * lognorm_amp_dist
    # integral over true_amp_mesh
    integral_amp = cp.trapz(mult_amp, true_amp_mesh, axis=1)    # print("first xlim",xlim)

    p_det = p_detect_cupy(amp)
    likelihood = integral_amp * p_det[:,0]
    integral = cp.trapz(likelihood, amp[:, 0])
    integral = 1-integral
    return cp.log(integral)*(N-n)

def first_cupy(amp,mu,std,xlim=20,x_len=1000,a=0,lower_c=0,upper_c=cp.inf):
    #xlim needs to be at least as large as 5 sigma_snrs though
    sigma_amp = det_error
    # amp is the detected amps
    # width is the detected widths
    # make an array of lower and upper limits for the true_log amp array
    sigma_lim = 5
    true_lower = amp - sigma_lim * sigma_amp
    true_lower[true_lower < 0] = cp.exp(-20)
    true_upper = amp + sigma_lim * sigma_amp
    # generate a mesh of amps
    # true_amp_mesh = cp.zeros((len(amp), x_len))
    true_amp_mesh = cp.linspace(cp.log(true_lower),cp.log(true_upper),x_len).T

    # for i, (l, u) in enumerate(zip(true_lower, true_upper)):
        # true_amp_mesh[i, :] = cp.linspace(cp.log(l), cp.log(u), x_len)
    amp = amp[:, cp.newaxis]
    gaussian_error_amp = gaussian_cupy(amp, cp.exp(true_amp_mesh), sigma_amp)
    lognorm_amp_dist = gaussian_cupy(true_amp_mesh, mu, std)
    mult_amp = gaussian_error_amp * lognorm_amp_dist
    # integral over true_amp_mesh
    integral_amp = cp.trapz(mult_amp, true_amp_mesh, axis=1)    # print("first xlim",xlim)
    p_det = p_detect_cupy(amp[:, 0])
    likelihood = integral_amp * p_det
    return cp.sum(cp.log(likelihood))
#################CUPY END#####################


def lognorm_dist(x, mu, sigma, lower_c=0, upper_c=np.inf):
    #lower and upper cutoff parameters added
    pdf = np.zeros(x.shape)
    mask = (x > lower_c) & (x < upper_c)
    pdf[mask] = np.exp(-((np.log(x[mask]) - mu) ** 2) / (2 * sigma**2)) / (
        (x[mask]) * sigma * np.sqrt(2 * np.pi)
    )
    def argument(c,mu,sigma):
        if c==0:
            return -np.inf
        return (np.log(c)-mu)/(sigma*np.sqrt(2))
    pdf = 2*pdf / (scipy.special.erf(argument(upper_c,mu,sigma))-scipy.special.erf(argument(lower_c,mu,sigma)))
    return pdf

def first_plot(amp,mu,std, sigma_snr=0.4, a=0, lower_c=0, upper_c=np.inf):
    x_len = 1000
    xlim = 500
    x_lims = [-xlim,xlim]
    amp_arr = np.linspace(x_lims[0],x_lims[1],x_len)
    gaussian_error = norm.pdf(amp_arr,0,sigma_snr)
    LN_dist = lognorm_dist(amp_arr,mu,std,lower_c=lower_c,upper_c=upper_c)
    #convolve the two arrays
    conv = np.convolve(LN_dist,gaussian_error)*np.diff(amp_arr)[0]
    conv_lims = [-xlim*2,xlim*2]
    conv_amp_array = np.linspace(conv_lims[0],conv_lims[1],(x_len*2)-1)+a
    #interpolate the values for amp
    p_det = p_detect_cpu(conv_amp_array)
    likelihood_conv = conv*p_det
    likelihood = np.interp(amp,conv_amp_array,likelihood_conv)
    return likelihood, p_det, conv_amp_array, conv

def total_p(X,snr_arr=None,use_a=False,use_cutoff=True,xlim=100,cuda_device=0):
    # print("starting loglike")
    with cp.cuda.Device(cuda_device):
        start = time.time()
        snr_arr = cp.array(snr_arr)
        transfer_time = time.time()
        mu = X["mu"]
        std = X["std"]
        N = X["N"]
        if use_a:
            a = X["a"]
        else:
            a = 0
        if use_cutoff:
            lower_c = X["lower_c"]
            upper_c = X["upper_c"]
        else:
            lower_c = 0
            upper_c = np.inf
        if lower_c>upper_c:
            print("lower_c is greater than upper_c")
            return -np.inf
        if snr_arr is None:
            snr_arr = X["snr_arr"]
        if N < len(snr_arr):
            raise Exception(" N<n")

        # print(f"mu: {mu}, std: {std}, N: {N}, a: {a}, lower_c: {lower_c}, upper_c: {upper_c}")
        sigma_snr = det_error
        f = first_cupy(snr_arr, mu, std,a=a,xlim=xlim,lower_c=lower_c,upper_c=upper_c)
        first_time = time.time()
        # print("finished f")
        if cp.isnan(f):
            print("f is nan")
            return -np.inf
        # s = second(len(snr_arr), mu, std, N, sigma_snr=sigma_snr)
        s = second_cupy(len(snr_arr), mu, std, N,a=a,xlim=xlim,lower_c=lower_c,upper_c=upper_c)
        second_time = time.time()
        # print("finished s")
        if cp.isnan(s):
            print("s is nan")
            return -np.inf
        n = len(snr_arr)
        log_NCn = cupy_gammaln(N + 1) - cupy_gammaln(n + 1) - cupy_gammaln(N - n + 1)
        # print("finished log_NCn")
        loglike = f + s + log_NCn
        loglike = np.array(loglike.get())
        overall_time = time.time()
        #print(f"transfer time: {transfer_time-start}, f time: {first_time-transfer_time}, s time: {second_time-first_time}, overall time: {overall_time-start}")
        # print(f"f: {f}, s: {s}, log_NCn: {log_NCn} loglike: {loglike}")
    return loglike

def negative_loglike(X, det_snr):
    x = {"mu": X[0], "std": X[1], "N": X[2], "snr_arr": det_snr}
    return -1 * total_p(x)

def mean_var_to_mu_std(mean, var):
    mu = np.log(mean**2/np.sqrt(var+mean**2))
    std = np.sqrt(np.log(var/mean**2+1))
    return mu, std

def mu_std_to_mean_var(mu, std):
    mean = np.exp(mu+std**2/2)
    var = (np.exp(std**2)-1)*np.exp(2*mu+std**2)
    return mean,var

def likelihood_lognorm(mu_arr, std_arr, N_arr, det_snr, mesh_size=20):
    # # create a mesh grid of N, mu and stds
    mat = np.zeros((len(mu_arr), len(std_arr), len(N_arr)))
    if max(det_snr)>100:
        xlim = max(det_snr)*2
    else:
        xlim = 100
    #with Pool(2) as po:
    X = []
    Y = []
    with cp.cuda.Device(0):
        det_snr = cp.array(det_snr)
    for i, mu_i in enumerate(mu_arr):
        for j, std_i in enumerate(std_arr):
            for k, N_i in enumerate(N_arr):
                #change the mu to a different definition
                mean_i, var_i = mu_std_to_mean_var(mu_i, std_i)
                upper_c = mean_i * 50
                X.append({"mu": mu_i, "std": std_i, "N": N_i, "snr_arr": det_snr, "lower_c": 0, "upper_c": upper_c})
                Y.append([mu_i,std_i,N_i])
    Y = np.array(Y)
    # m = np.array(po.map(total_p, X))
    m = []
    for ind,v in enumerate(X):
        print(f"{ind}/{len(X)}")
        m.append(total_p(v,det_snr,use_cutoff=True,cuda_device=0,xlim=xlim))
    m = np.array(m)
    for i, mu_i in enumerate(mu_arr):
        for j, std_i in enumerate(std_arr):
            for k, N_i in enumerate(N_arr):
                ind = np.sum((Y==[mu_i,std_i,N_i]),axis=1)==3
                mat[i,j,k] = m[ind]

    return mat
