#!/usr/bin/env python3

import dill
import numpy as np
import sys
from simulate_pulse import simulate_pulses
from simulate_pulse import n_detect
from simulate_pulse import simulate_pulses_exp
import matplotlib.pyplot as plt

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
    parser.add_argument('-e', action="store_true", default=False,help="turn flag on for exponential distribution")





    args = parser.parse_args()
    mu = args.mu
    std = args.std
    obs_t = args.obs
    p = args.p
    f = args.f
    dill_file = args.d
    from numpy.random import normal
    sigma_snr = 1
    if args.e:
        pulses = simulate_pulses_exp(obs_t,p,f,mu)
    else:
        pulses = simulate_pulses(obs_t,p,f,mu,std)
    rv = normal(loc=0,scale=sigma_snr,size=len(pulses))
    pulses = rv+pulses
    detected_pulses = n_detect(pulses)
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

    plt.figure()
    plt.hist(detected_pulses,bins= 100,density=True,label="original fake data")
    plt.hist(det_snr,bins=100,density=True,alpha=0.5,label="new fake data")
    plt.legend()
    plt.show()
