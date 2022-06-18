#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

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
    pulse_snr = simulate_pulses_exp(50000,2,1,k=4)
    plt.hist(pulse_snr,bins=1000)
    plt.show()
