#!/usr/bin/env python3
from scipy.special import gammaln
from multiprocessing import Pool
import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import exponnorm
from scipy.stats import norm
import dill
from scipy.stats import expon
from cupyx.scipy.special import gammaln as cupy_gammaln
import time
import statistics_basic
from statistics_basic import load_detection_fn, p_detect, p_detect_cupy
global det_error
det_error = statistics_basic.det_error
print("det_error for exp",det_error)

###############################CUPY FUNCTIONS##################################
import cupy as cp
def exponential_dist_cupy(x, k):
    pdf = cp.zeros(x.shape)
    pdf[x > 0] = k * cp.exp(-k * x[x > 0])
    return pdf

def gaussian_cupy(x, mu, sigma):
    return cp.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * cp.sqrt(2 * cp.pi))

def second_exp_cupy(n,k,N,xlim=110,x_len=5000000):
     #xlim needs to be at least as large as 5 sigma_snrs though
    wide_enough = False
    sigma_snr = det_error
    x_lims = [-sigma_snr*10,xlim]
    amp_arr = cp.linspace(x_lims[0],x_lims[1],x_len)
    dist = exponential_dist_cupy(amp_arr,k)
    gaussian_error = gaussian_cupy(amp_arr,0,sigma_snr)

    # print("second xlim",xlim)
    #convolve the two arrays
    # plt.figure()
    # plt.plot(amp_arr,LN_dist)
    # plt.plot(amp_arr,gaussian_error)
    # integral_ln = np.trapz(LN_dist,amp_arr)
    # plt.title(f"xlen {x_len} xlim {xlim} integral {integral_ln}")
    conv = cp.convolve(dist,gaussian_error)*cp.diff(amp_arr)[0]
    conv_lims = [2*x_lims[0],2*x_lims[1]]
    conv_amp_array = cp.linspace(conv_lims[0],conv_lims[1],(x_len*2)-1)
    #interpolate the values for amp
    p_det = p_detect_cupy(conv_amp_array)
    likelihood = conv*(1-p_det)
    # likelihood = np.interp(amp,conv_amp_array,likelihood_conv)
    integral = cp.trapz(likelihood,conv_amp_array)
    return cp.log(integral)*(N-n)

def first_exp_cupy(amp,k,xlim=100,x_len=100000):
    #xlim needs to be at least as large as 5 sigma_snrs though
    sigma_snr = det_error
    x_lims = [-xlim,xlim]
    amp_arr = cp.linspace(x_lims[0],x_lims[1],x_len)
    dist = exponential_dist_cupy(amp_arr,k)
    gaussian_error = gaussian_cupy(amp_arr,0,sigma_snr)
    # print("first xlim",xlim)
    #convolve the two arrays
    conv = cp.convolve(dist,gaussian_error)*cp.diff(amp_arr)[0]
    conv_lims = [2*x_lims[0],2*x_lims[1]]
    conv_amp_array = cp.linspace(conv_lims[0],conv_lims[1],(x_len*2)-1)
    #interpolate the values for amp
    p_det = p_detect_cupy(conv_amp_array)
    likelihood_conv = conv*p_det
    likelihood = cp.interp(amp,conv_amp_array,likelihood_conv)
    return cp.sum(cp.log(likelihood))



#################CUPY END#####################



def k_to_mean_var(k):
    return 1/k,1/k**2

def first_exp_plot(amp,k, sigma_snr=0.4):
    x_len = 10000
    xlim= 200/k
    x_lims = [-xlim,xlim]
    amp_arr = np.linspace(x_lims[0],x_lims[1],x_len)
    gaussian_error = norm.pdf(amp_arr,0,sigma_snr)
    LN_dist = expon.pdf(amp_arr,scale=1/k)
    #convolve the two arrays
    conv = np.convolve(LN_dist,gaussian_error)*np.diff(amp_arr)[0]
    conv_lims = [-xlim*2,xlim*2]
    conv_amp_array = np.linspace(conv_lims[0],conv_lims[1],(x_len*2)-1)
    #interpolate the values for amp
    p_det = p_detect(conv_amp_array)
    # p_det = inj_stats.predict_poly(conv_amp_array,inj_stats.detected_bin_midpoints,inj_stats.detected_det_frac)

    likelihood_conv = conv*p_det
    likelihood = np.interp(amp,conv_amp_array,likelihood_conv)
    # plt.close()
    # plt.hist(amp,density=True)
    # plt.plot(conv_amp_array,likelihood_conv/np.trapz(likelihood_conv,conv_amp_array))
    # plt.plot(amp_arr,LN_dist)
    # plt.show()
    # import pdb; pdb.set_trace()
    return likelihood

def first_exp(amp,k, sigma_snr=0.4):
    x_len = 10000
    xlim= 200/k
    if xlim<5*sigma_snr:
        xlim = 5*sigma_snr
    x_lims = [-xlim,xlim]
    amp_arr = np.linspace(x_lims[0],x_lims[1],x_len)
    gaussian_error = norm.pdf(amp_arr,0,sigma_snr)
    LN_dist = expon.pdf(amp_arr,scale=1/k)
    #convolve the two arrays
    conv = np.convolve(LN_dist,gaussian_error)*np.diff(amp_arr)[0]
    conv_lims = [-xlim*2,xlim*2]
    conv_amp_array = np.linspace(conv_lims[0],conv_lims[1],(x_len*2)-1)
    #interpolate the values for amp
    p_det = p_detect(conv_amp_array)
    # p_det = inj_stats.predict_poly(conv_amp_array,inj_stats.detected_bin_midpoints,inj_stats.detected_det_frac)

    likelihood_conv = conv*p_det
    likelihood = np.interp(amp,conv_amp_array,likelihood_conv)
    # plt.close()
    # plt.hist(amp,density=True)
    # plt.plot(conv_amp_array,likelihood_conv/np.trapz(likelihood_conv,conv_amp_array))
    # plt.plot(amp_arr,LN_dist)
    # plt.show()
    # import pdb; pdb.set_trace()
    return np.sum(np.log(likelihood))

def second_exp(n, k, N, sigma_snr):

    x_len = 10000
    xlim=200/k
    if xlim<5*sigma_snr:
        xlim = 5*sigma_snr
    x_lims = [-xlim,xlim]
    amp = np.linspace(-xlim / 2, xlim / 2, 10000)

    amp_arr = np.linspace(x_lims[0],x_lims[1],x_len)
    gaussian_error = norm.pdf(amp_arr,0,sigma_snr)
    # p_not_det = 1-inj_stats.predict_poly(amp,inj_stats.detected_bin_midpoints,inj_stats.detected_det_frac)
    p_not_det = 1-p_detect(amp)
    # p_det_giv_param = p_detect(amp_arr)*norm.pdf(amp_arr,mu,std)
    LN_dist = expon.pdf(amp_arr,scale=1/k)
    # plt.figure()
    # plt.plot(amp_arr,LN_dist)
    # plt.show()
    #convolve the two arrays
    conv = np.convolve(LN_dist,gaussian_error)*np.diff(amp_arr)[0]
    conv_lims = [-xlim*2,xlim*2]
    conv_amp_array = np.linspace(conv_lims[0],conv_lims[1],(x_len*2)-1)
    #interpolate the values for amp
    # p_det = inj_stats.predict_poly(conv_amp_array,inj_stats.detected_bin_midpoints,inj_stats.detected_det_frac)
    p_det = p_detect(conv_amp_array)
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

def total_p_exp(X,snr_arr=None,use_a=False,use_cutoff=True,xlim=100,cuda_device=0):
    with cp.cuda.Device(cuda_device):
        start = time.time()
        snr_arr = cp.array(snr_arr)
        transfer_time = time.time()
        k = X["k"]
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

        sigma_snr = cp.array(det_error)
        f = first_exp_cupy(snr_arr, k)
        if cp.isnan(f):
            print("f is nan")
            return -cp.inf
        s = second_exp_cupy(len(snr_arr), k, N)
        if cp.isnan(s):
            print("s is nan")
            return -cp.inf
        n = len(snr_arr)
        log_NCn = cupy_gammaln(N + 1) - cupy_gammaln(n + 1) - cupy_gammaln(N - n + 1)
        # print(k,N)
        # print(f,s,log_NCn)
        loglike = f + s + log_NCn
        loglike = np.array(loglike.get())
    return loglike

def negative_loglike(X, det_snr):
    x = {"k": X[0], "N": X[1], "snr_arr": det_snr}
    return -1 * total_p_exp(x)

def likelihood_exp(k_arr, N_arr, det_snr):
    mat = np.zeros((len(k_arr), len(N_arr)))

    with Pool(10) as po:
        X = []
        Y = []
        for i, k_i in enumerate(k_arr):
            for j, N_i in enumerate(N_arr):
                X.append({"k": k_i, "N": N_i, "snr_arr": det_snr})
                Y.append([k_i,N_i])
        Y = np.array(Y)
        # m = np.array(po.map(total_p_exp, X))
        m = []
        for ind,v in enumerate(X):
            print(f"{ind}/{len(X)}")
            m.append(total_p_exp(v))
        m = np.array(m)

        for i, k_i in enumerate(k_arr):
                for j, N_i in enumerate(N_arr):
                    ind = np.sum((Y==[k_i,N_i]),axis=1)==2
                    mat[i,j] = m[ind]
    return mat

