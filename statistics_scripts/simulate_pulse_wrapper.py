#!/usr/bin/env python3

import dill
import numpy as np
import sys
from simulate_pulse import simulate_pulses
from simulate_pulse import n_detect
from simulate_pulse import simulate_pulses_exp
from simulate_pulse import simulate_pulses_gauss
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import expon
from scipy.stats import truncnorm
from statistics import lognorm_dist
import dill
from scipy.integrate import quad
with open("inj_stats_fitted.dill", "rb") as inf:
    inj_stats = dill.load(inf)
# popt = inj_stats.fit_logistic_amp
det_error = inj_stats.detect_error_amp
sigma_snr=det_error

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

def time_varying_convolve(f, g, t, *args):
    """
    Compute the time-varying convolution of two functions.

    Parameters:
        f (callable): The function to be convolved.
        g (callable): The kernel function that is changing over time.
        t (array_like): The array of time values over which to compute the convolution.
        *args: Additional arguments to be passed to `f` and `g`.

    Returns:
        array_like: The result of the convolution at each time value.
    """
    result = np.zeros_like(t)
    for i, ti in enumerate(t):
        integrand = lambda u: f(*args, ti-u) * g(u)
        result[i], _ = quad(integrand, -np.inf, np.inf)
    return result

def g(amp_arr):
    shifted_amp_array = amp_array/logistic(amp_arr,inj_stats.error_correction_log_params[0],inj_stats.error_correction_log_params[1])
    norm.pdf(shifted_amp_array,0)

def f(mu,std,amp_arr):
    #this is the function that describes the probability density of the _true_ emission amplitude
    return inj_stats.predict_poly(amp_arr)*lognorm_dist(amp_arr,mu,std)

def convolve_first(mu_snr,mu,std, sigma_snr=0.4):
    x_len = 1000000
    const = 91
    a = -mu/std
    b = (100-mu)/std

    # xlim = (100/mu)
    # print(xlim)
    xlim = np.exp(mu)*std*const
    x_lims = [-xlim, xlim]
    snr_arr = np.linspace(x_lims[0],x_lims[1],x_len)
    shift_function = logistic(snr_arr,inj_stats.error_correction_log_params[0],inj_stats.error_correction_log_params[1])
    p_musnr_giv_snr = norm2(snr_arr/shift_function,0,sigma_snr)
    # p_det_giv_param = inj_stats.predict_poly(snr_arr)*expon.pdf(snr_arr,scale=1/mu)
    # p_det_giv_param = inj_stats.predict_poly(snr_arr)*truncnorm.pdf(snr_arr,loc=mu,scale=std,a=a,b=b)
    p_det_giv_param = inj_stats.predict_poly(snr_arr)*lognorm_dist(snr_arr,mu,std)
    #convolve the two arrays

    conv = np.convolve(p_musnr_giv_snr,p_det_giv_param)*np.diff(snr_arr)[0]
    conv_lims = [-(xlim*2), xlim*2]
    conv_snr_array = np.linspace(conv_lims[0],conv_lims[1],(x_len*2)-1)
    # shifted_conv_snr_array = logistic(conv_snr_array,inj_stats.error_correction_log_params[0],inj_stats.error_correction_log_params[1]) * conv_snr_array

    #interpolate the values for mu_snr
    convolve_mu_snr = np.interp(mu_snr,conv_snr_array,conv)

    plt.close()
    plt.figure()
    plt.plot(conv_snr_array,conv,label="conv",linewidth=5)
    plt.plot(snr_arr,p_musnr_giv_snr,alpha=0.5,label="gauss2")
    plt.plot(snr_arr,p_det_giv_param,alpha=0.5,label="pdet")
    plt.scatter(mu_snr,convolve_mu_snr,alpha=0.5,label="interp",c='k')
    plt.legend()
    plt.show()
    return convolve_mu_snr


def norm2(snr_arr,mu=0,sigma_snr=0.4):
    return norm.pdf(snr_arr,loc=mu,scale=sigma_snr)


def first_gauss(snr,mu,std,sigma_snr=0.4):
    return convolve_first(snr,mu,std,sigma_snr)



if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Simulate some pulses')
    parser.add_argument('-mu', type=float, default=0.5,
                        help='mean or the k parameter for an exponential distribution')
    parser.add_argument('-std', type=float, default=0.2,
                        help='standard deviation')
    parser.add_argument('-obs', type=float, default=1000,
                        help='standard deviation')
    parser.add_argument('-p', type=float, default=1,
                        help='standard deviation')
    parser.add_argument('-f', type=float, default=1,
                        help='standard deviation')
    parser.add_argument('-d', type=str, default="",
                        help='dill_file')
    parser.add_argument('--mode', type=str, default="Lognorm",help="Choose the distribution you want the fake data to have, the options are Lognorm, Gauss, Exp")





    args = parser.parse_args()
    mu = args.mu
    std = args.std
    obs_t = args.obs
    p = args.p
    f = args.f
    dill_file = args.d
    from numpy.random import normal
    snr = np.linspace(0,0.4,100)
    det_fn = inj_stats.predict_poly(snr)
    plt.plot(snr,det_fn)
    plt.show()
    print("sigma_snr",sigma_snr)

    #save the detection function with the detection error


    mode = args.mode
    if mode=="Exp":
        pulses = simulate_pulses_exp(obs_t,p,f,mu)
    elif mode=="Lognorm":
        pulses = simulate_pulses(obs_t,p,f,mu,std)
    elif mode=="Gauss":
        pulses,a,b = simulate_pulses_gauss(obs_t,p,f,mu,std)

    # rv = normal(loc=0,scale=sigma_snr,size=len(pulses))
    # pulses = rv+pulses
    detected_pulses = n_detect(pulses)
    #rv is a biased gaussian
    conv_factor = logistic(detected_pulses,inj_stats.error_correction_log_params[0],inj_stats.error_correction_log_params[1])
    means = (detected_pulses/conv_factor) - detected_pulses
    stds = np.zeros(len(means))+sigma_snr
    rv = normal(loc=means,scale=stds)
    detected_pulses = rv+detected_pulses
    print(len(detected_pulses))
    import dill
    with open(dill_file,'rb') as inf:
        det_class = dill.load(inf)

    det_snr = []
    for pulse_obj in det_class.sorted_pulses:
        if pulse_obj.det_snr!=-1:
            #-1 means that the snr could not be measured well
            det_snr.append(pulse_obj.det_snr)

    plt.figure()
    plt.hist(detected_pulses,bins= "auto",density=True,label="fake data")
    plt.hist(pulses,bins= "auto",density=True,alpha=0.6,label="fake data no selection")
    plt.hist(det_snr,bins="auto",density=True,alpha=0.5,label="real data")
    plt.legend()
    plt.show()
    #create a fake det_classes
    filfiles = np.full(len(detected_pulses),"abc",dtype=str)
    maskfn = np.full(len(detected_pulses),"abc",dtype=str)
    dms = np.full(len(detected_pulses),123,dtype=float)
    toas = np.full(len(detected_pulses),123,dtype=float)
    from inject_stats import inject_obj
    inject_obj_arr = []
    for snr in detected_pulses:
        temp = inject_obj()
        temp.det_snr = snr
        temp.det_fluence = snr
        temp.det_amp = snr
        temp.fluence_amp = snr
        temp.det_std = snr
        inject_obj_arr.append(temp)
    inj_obj_arr = np.array(inject_obj_arr)
    det_class.filfiles = filfiles
    det_class.mask_fn = maskfn
    det_class.dms = dms
    det_class.toas = toas
    det_class.sorted_pulses = inject_obj_arr
    with open('fake_data.dill','wb') as of:
        dill.dump(det_class,of)

    det_snr = []
    for pulse_obj in det_class.sorted_pulses:
        if pulse_obj.det_snr!=-1:
            #-1 means that the snr could not be measured well
            det_snr.append(pulse_obj.det_snr)

    snr_array = np.linspace(-1,1,10000)
    first = first_gauss(snr_array,mu,std,sigma_snr)
    first = first/np.trapz(first,snr_array)
    plt.figure()
    plt.hist(det_snr,bins="auto",density=True,alpha=0.5,label="new fake data")
    plt.plot(snr_array,first)
    plt.legend()
    plt.show()
