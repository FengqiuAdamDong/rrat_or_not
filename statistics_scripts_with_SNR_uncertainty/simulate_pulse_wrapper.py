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
import statistics_basic
import dill
import argparse

parser = argparse.ArgumentParser(description='Simulate some pulses')
parser.add_argument('-mu', type=float, default=0.5,
                    help='mean or the k parameter for an exponential distribution')
parser.add_argument('-std', type=float, default=0.2,
                    help='standard deviation')
parser.add_argument('-obs', type=float, default=10000,
                    help='standard deviation')
parser.add_argument('-p', type=float, default=1,
                    help='standard deviation')
parser.add_argument('-f', type=float, default=1,
                    help='standard deviation')
parser.add_argument('-d', type=str, default="",
                    help='dill_file')
parser.add_argument('-inj_file', type=str,help="injection_file.dill")
parser.add_argument('--mode', type=str, default="Lognorm",help="Choose the distribution you want the fake data to have, the options are Lognorm, Gauss, Exp")

args = parser.parse_args()
mu = args.mu
std = args.std
obs_t = args.obs
p = args.p
f = args.f
dill_file = args.d
inj_file = args.inj_file

statistics_basic.load_detection_fn(inj_file)

from statistics import lognorm_dist

if __name__=='__main__':

    from numpy.random import normal
    import statistics
    statistics.load_detection_fn(inj_file)
    sigma_snr = statistics.det_error
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
        detected_pulses = n_detect(d_pulses,inj_file)
        #rv is a biased gaussian
        # print("len detected",len(detected_pulses))
        n.append(len(detected_pulses))

    print("generated",len(pulses))
    plt.hist(n,bins="auto")
    plt.xlabel("Number of pulses detected")
    print(len(detected_pulses))

    import dill
    with open(dill_file,'rb') as inf:
        det_class = dill.load(inf)

    det_snr = []
    for pulse_obj in det_class.sorted_pulses:
        if pulse_obj.det_snr!=-1:
            #-1 means that the snr could not be measured well
            det_snr.append(pulse_obj.det_snr)

    # plt.figure()
    # plt.hist(detected_pulses,bins= "auto",density=True,label="fake data")
    # plt.hist(pulses,bins= "auto",density=True,alpha=0.6,label="fake data no selection")
    # plt.hist(det_snr,bins="auto",density=True,alpha=0.5,label="real data")
    # plt.legend()
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
    if mode=="Lognorm":
        p_dist = statistics.first_plot(snr_array,mu,std,sigma_snr)
    elif mode=="Exp":
        import statistics_exp
        print(statistics_exp.det_error)
        p_dist = statistics_exp.first_exp_plot(snr_array,mu,sigma_snr)

    p_dist = p_dist/np.trapz(p_dist,snr_array)
    plt.figure()
    plt.hist(det_snr,bins="auto",density=True,alpha=0.5,label="new fake data")
    plt.plot(snr_array,statistics.p_detect(snr_array),label="det_prob")
    plt.plot(snr_array,lognorm_dist(snr_array,mu,std),label='lognorm')
    plt.plot(snr_array,p_dist,label='detection function')
    plt.legend()
    plt.show()