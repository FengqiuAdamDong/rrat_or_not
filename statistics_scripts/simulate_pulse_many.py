#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os
def p_detect(snr,k=5.5762911,x0=2.12284101,L=1):
    #this will just be an exponential rise at some center
    #added a decay rate variable just so things are compatible
    return L/(1+np.exp(-k*(snr-x0)))

def n_detect(snr_emit):
    #snr emit is the snr that the emitted pulse has
    p = p_detect(snr_emit)
    #simulate random numbers between 0 and 1
    rands = np.random.rand(len(p))
    #probability the random number is less than p gives you an idea of what will be detected
    detected = snr_emit[rands<p]
    return detected

#so in this script we need to simulate N pulses from a pulsar
def simulate_pulses(obs_t,period,pulse_p,mu,std):
    #number of pulses
    N = int(obs_t/period)
    #draw N random variables between 0 and 1
    rands = np.random.rand(N)
    #check how many are successes
    pulse_N = np.sum(rands<pulse_p)
    pulse_snr = np.random.normal(mu,std,pulse_N)
    return 10**pulse_snr

def simulate_pulses_exp(obs_t,period,pulse_p,k):
    #we simulate the pulses as a power law instead of a log normal
    #number of pulses
    N = int(obs_t/period)
    #draw N random variables between 0 and 1
    rands = np.random.rand(N)
    #cdf = 1-np.exp(-k*x)
    pulse_N = np.sum(rands<pulse_p)
    rands = np.random.rand(pulse_N)
    pulse_snr = -np.log(1-rands)/k
    return pulse_snr

if __name__=='__main__':
    obs_t =5000000
    mu = -0.1
    std = 0.1
    # k_arr = np.linspace(0.01,1,1000)
    p = 2
    frac = 0.1
    save_folder = 'bayes_sims'
    #simulate the pulses for log-norm
    for i in range(100):
        pulse_snr = simulate_pulses(obs_t,p,frac,mu,std)
        det_snr = n_detect(pulse_snr)
        print(len(det_snr))
        fn = f"mu_{mu}_std_{std}"
        save_dir = f"{save_folder}_ln_{mu}_{std}_{frac}"
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        fn = f"{save_dir}/{fn}_{i}"
        np.savez(fn,snrs=pulse_snr,det=det_snr,true_mu=mu,true_std=std,p=p,true_frac=frac,obs_t=obs_t)
    # for k in k_arr:
    #     pulse_snr = simulate_pulses_exp(obs_t,p,frac,k)
    #     det_snr = n_detect(pulse_snr)
    #     fn = f"k_{k}"
    #     save_dir = f"{save_folder}_exp"
    #     # plt.figure()
    #     # plt.hist(pulse_snr)
    #     # plt.figure()
    #     # plt.hist(det_snr)
    #     # plt.show()
    #     print(len(det_snr))
    #     if not os.path.isdir(save_dir):
    #         os.mkdir(save_dir)
    #     fn = f"{save_dir}/{fn}"
    #     np.savez(fn,snrs=pulse_snr,det=det_snr,true_k=k,p=p,true_frac=frac,obs_t=obs_t)
