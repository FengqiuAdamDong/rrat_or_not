#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os
from statistics import p_detect

def n_detect(snr_emit):
    #snr emit is the snr that the emitted pulse has
    p = p_detect(snr_emit)
    #simulate random numbers between 0 and 1
    rands = np.random.rand(len(p))
    #probability the random number is less than p gives you an idea of what will be detected
    detected = snr_emit[rands<p]
    return detected

#so in this script we need to simulate N pulses from a pulsar
def simulate_pulses(obs_t,period,f,mu,std,random=True):
    #number of pulses
    N = int(obs_t/period)
    #draw N random variables between 0 and 1
    rands = np.random.rand(N)
    #check how many are successes
    if random:
        pulse_N = np.sum(rands<f)
    else:
        pulse_N = int(N*f)
    pulse_snr = np.random.lognormal(mu,std,pulse_N)
    return pulse_snr

def simulate_pulses_exp(obs_t,period,f,k,random=True):
    #we simulate the pulses as a power law instead of a log normal
    #number of pulses
    N = int(obs_t/period)
    #draw N random variables between 0 and 1
    rands = np.random.rand(N)
    #cdf = 1-np.exp(-k*x)
    if random:
        pulse_N = np.sum(rands<f)
    else:
        pulse_N = int(N*f)
    rands = np.random.rand(pulse_N)
    pulse_snr = -np.log(1-rands)/k
    return pulse_snr

if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Simulate some pulses')
    parser.add_argument('-N', type=int,
                        help='Detected pulse number if this is given along with a distribution type then p and observation time parameters are not required')
    parser.add_argument('-p', type=float, default=2,
                        help='Pulsar period')
    parser.add_argument('-O', type=float, default=500000,
                        help='Observation time (s)')
    parser.add_argument('-ln', default=True,action='store_false',
                        help='Use the log normal distribution (otherwise use exponential)')
    parser.add_argument('-props', type=float, nargs='+',
                        help='properties of the distribution give 2 parameters of LN 1 parameters of exponential (Mu, sigma) and (k)')
    parser.add_argument('-f', type=float, default=0.001,
                        help='Pulse fraction')
    parser.add_argument('-od', type=str, default='bayes_sim',
                        help='The output directory')
    parser.add_argument('-iterations', type=int, default='1',
                        help='How many simulates to generate')


    args = parser.parse_args()
    loop = args.iterations
    if not args.N:
        obs_t = args.O
        p = args.p
        frac = args.f
    else:
        N = args.N
        obs_t_orig = 10000
        p = 2
        frac = 0.01
    save_folder = args.od
    for i in range(args.iterations):
        if args.ln:
            mu,std = args.props
            if N:
                pulse_snr = []
                det_snr = []
                obs_t = obs_t_orig
                while len(det_snr)<N:
                    obs_t = obs_t+(p*200/frac)
                    pulse_snr = simulate_pulses(obs_t,p,frac,mu,std,random=False)
                    det_snr = n_detect(pulse_snr)
            else:
                pulse_snr = simulate_pulses(obs_t,p,frac,mu,std,random=False)
                det_snr = n_detect(pulse_snr)
            print(f"iter {i}: {len(det_snr)}")
            fn = f"mu_{mu}_std_{std}_{i}"
            save_dir = f"{save_folder}_ln_{mu}_{std}_{N}"
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            fn = f"{save_dir}/{fn}"
            np.savez(fn,snrs=pulse_snr,det=det_snr,true_mu=mu,true_std=std,p=p,true_frac=frac,obs_t=obs_t)
        else:
            k = args.props[0]
            if N:
                pulse_snr = []
                det_snr = []
                obs_t = obs_t_orig
                while len(det_snr)<N:
                    obs_t = obs_t+(p/frac)
                    pulse_snr = simulate_pulses_exp(obs_t,p,frac,k,random=False)
                    det_snr = n_detect(pulse_snr)
            else:
                pulse_snr = simulate_pulses_exp(obs_t,p,frac,k,random=False)
                det_snr = n_detect(pulse_snr)
            fn = f"k_{k}_{i}"
            save_dir = f"{save_folder}_exp_{k}"
            print(len(det_snr))
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            fn = f"{save_dir}/{fn}"
            np.savez(fn,snrs=pulse_snr,det=det_snr,true_k=k,p=p,true_frac=frac,obs_t=obs_t)
