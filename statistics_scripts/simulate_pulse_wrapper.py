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
from statistics import lognorm_dist

def logistic(x,k,x0):
    L=1
    return L/(1+np.exp(-k*(x-x0)))

def convolve_first(mu_snr,mu,std, sigma_snr=0.4):
    x_len = 10000
    x_lims = [-20,20]
    snr_arr = np.linspace(x_lims[0],x_lims[1],x_len)
    p_musnr_giv_snr = norm2(snr_arr,0,sigma_snr)
    # p_det_giv_param = p_detect(snr_arr)*norm.pdf(snr_arr,mu,std)
    p_det_giv_param = p_detect(snr_arr)*lognorm_dist(snr_arr,mu,std)
    #convolve the two arrays
    conv = np.convolve(p_det_giv_param,p_musnr_giv_snr)*np.diff(snr_arr)[0]
    conv_lims = [-40,40]
    conv_snr_array = np.linspace(conv_lims[0],conv_lims[1],(x_len*2)-1)
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


def p_detect(snr,cutoff=1):
    #this will just be an exponential rise at some center
    #added a decay rate variable just so things are compatible
    #load inj statistics
    k = popt[0]
    x0 = popt[1]
    # print(k,x0)
    L = 1
    detection_fn = L/(1+np.exp(-k*(snr-x0)))
    # detection_fn[snr<cutoff] = 0

    return detection_fn

# def first_gauss(snr,mu,std,sigma_snr=0.4):
#     #adding two gaussians
#     comb_std = np.sqrt(std**2+sigma_snr**2)
#     # comb_std = std
#     expmodnorm = norm.pdf(snr,loc=mu,scale=comb_std)
#     p_det = p_detect(snr,sigma_snr)
#     return p_det*expmodnorm

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
    popt = np.load("det_fun_params.npz", allow_pickle=1)["popt"]
    sigma_snr = np.load("det_fun_params.npz", allow_pickle=1)["det_error"]
    print(sigma_snr)
    #save the detection function with the detection error


    mode = args.mode
    if mode=="Exp":
        pulses = simulate_pulses_exp(obs_t,p,f,mu)
    elif mode=="Lognorm":
        pulses = simulate_pulses(obs_t,p,f,mu,std)
    elif mode=="Gauss":
        pulses = simulate_pulses_gauss(obs_t,p,f,mu,std)

    # rv = normal(loc=0,scale=sigma_snr,size=len(pulses))
    # pulses = rv+pulses
    detected_pulses = n_detect(pulses)
    rv = normal(loc=0,scale=sigma_snr,size=len(detected_pulses))
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

    snr_array = np.linspace(0,10,1000)
    first = first_gauss(snr_array,mu,std,sigma_snr)
    first = first/np.trapz(first,snr_array)
    plt.figure()
    plt.hist(det_snr,bins="auto",density=True,alpha=0.5,label="new fake data")
    plt.plot(snr_array,first)
    plt.legend()
    plt.show()
