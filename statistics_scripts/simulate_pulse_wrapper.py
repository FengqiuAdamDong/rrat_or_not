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
with open("inj_stats_fitted.dill", "rb") as inf:
    inj_stats = dill.load(inf)
# popt = inj_stats.fit_logistic_amp
det_error = inj_stats.detect_error_snr
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


def convolve_first(amp,mu,std, sigma_snr=0.4):
    x_len = 10000
    x_lims = [-10,10]
    amp_arr = np.linspace(x_lims[0],x_lims[1],x_len)
    gaussian_error = norm.pdf(amp_arr,0,sigma_snr)
    p_det = inj_stats.predict_poly(amp,inj_stats.detected_bin_midpoints,inj_stats.detected_det_frac)
    # p_det_giv_param = p_detect(amp_arr)*norm.pdf(amp_arr,mu,std)
    LN_dist = lognorm_dist(amp_arr,mu,std)
    # LN_dist = expon.pdf(amp_arr,scale=1/mu)
    #convolve the two arrays
    conv = np.convolve(LN_dist,gaussian_error)*np.diff(amp_arr)[0]
    conv_lims = [-20,20]
    conv_amp_array = np.linspace(conv_lims[0],conv_lims[1],(x_len*2)-1)
    #interpolate the values for amp
    convolve_amp = np.interp(amp,conv_amp_array,conv)
    likelihood = convolve_amp*p_det
    plt.close()
    plt.figure()
    plt.plot(conv_amp_array,conv,label="conv",linewidth=5)
    plt.plot(amp,likelihood,alpha=0.5,label="likelihood")

    plt.plot(amp,p_det,alpha=0.5,label="pdet")
    plt.scatter(amp,convolve_amp,alpha=0.5,label="interp",c='k')
    plt.legend()
    plt.show()
    return likelihood

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
    snr = np.linspace(0,5,10000)
    det_fn = inj_stats.predict_poly(snr,inj_stats.detected_bin_midpoints,inj_stats.detected_det_frac)
    plt.plot(snr,det_fn)
    plt.show()
    print("sigma_snr",sigma_snr)

    #save the detection function with the detection error
    n = []

    mode = args.mode
    if mode=="Exp":
        pulses = simulate_pulses_exp(obs_t,p,f,mu,random=False)
    elif mode=="Lognorm":
        pulses = simulate_pulses(obs_t,p,f,mu,std,random=False)
    elif mode=="Gauss":
        pulses,a,b = simulate_pulses_gauss(obs_t,p,f,mu,std,random=True)

    for i in range(1):
        rv = normal(loc=0,scale=sigma_snr,size=len(pulses))
        d_pulses = rv+pulses
        # print("len simulated",len(pulses))
        detected_pulses = n_detect(d_pulses)
        #rv is a biased gaussian
        # print("len detected",len(detected_pulses))
        n.append(len(detected_pulses))

    print("generated",len(pulses))
    plt.hist(n,bins="auto")
    plt.xlabel("Number of pulses detected")
    plt.show()
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
        temp.noise_std = snr
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

    snr_array = np.linspace(0,20,10000)
    # mu = -3
    # std = 0.4
    first = first_gauss(snr_array,mu,std,sigma_snr)
    first = first/np.trapz(first,snr_array)
    plt.figure()
    plt.hist(det_snr,bins="auto",density=True,alpha=0.5,label="new fake data")
    plt.plot(snr_array,first)
    plt.legend()
    plt.show()
