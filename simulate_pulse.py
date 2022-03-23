#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

#so in this script we need to simulate N pulses from a pulsar
#
def simulate_pulses(obs_t,period,pulse_p,mu,std):
    #number of pulses
    N = int(obs_t/period)
    #draw N random variables between 0 and 1
    rands = np.random.rand(N)
    #check how many are successes
    pulse_N = np.sum(rands<pulse_p)
    pulse_snr = np.random.normal(mu,std,pulse_N)
    return 10**pulse_snr

