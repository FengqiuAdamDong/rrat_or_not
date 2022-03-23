#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from simulate_pulse import simulate_pulses
from scipy.stats import norm
from math import comb

def p_detect(snr,decay_rate,lower_cutoff=6):
    #this will just be an exponential rise at some center
    p = 1-np.exp(-1*decay_rate*(snr-lower_cutoff))
    p[snr<lower_cutoff] = 0
    return p

def n_detect(snr_emit):
    #snr emit is the snr that the emitted pulse has
    p = p_detect(snr_emit,2)
    #simulate random numbers between 0 and 1
    rands = np.random.rand(len(p))
    #probability the random number is less than p gives you an idea of what will be detected
    detected = snr_emit[rands<p]
    return detected

def snr_distribution(snr,mu,std):
    #create meshgrids
    return norm.pdf(np.log10(snr),loc=mu,scale=std)

def first(snr,mu,std):
    p_det = p_detect(snr,2)
    snr_p = snr_distribution(snr,mu,std)
    return np.prod(p_det*snr_p)

def second(n,mu,std,N):
    snr_arr = np.linspace(0.1,100,10000)
    p_snr = snr_distribution(snr_arr,mu,std)
    p_not_det = 1-(p_detect(snr_arr,2))
    p_second = p_snr*p_not_det
    #integrate over flux
    p_second_int = np.trapz(p_second,np.log10(snr_arr))
    if p_second_int>1:
        # print(p_second_int)
        p_second_int=1
    return p_second_int**(N-n)

def total_p(mu,std,N,snr_arr):
    f = first(snr_arr,mu,std)
    s = second(len(snr_arr),mu,std,N)
    NCn = comb(N,len(snr_arr))
    # print(NCn,f,s)

    return NCn*f*s

pulse_snrs = simulate_pulses(1000,2,0.5,0.85,0.1)
det_snr = n_detect(pulse_snrs)
print(len(pulse_snrs),len(det_snr))
mesh_size = 100
#create a mesh grid of N, mu and stds
mu_arr = np.linspace(0.01,5,mesh_size)
std_arr = np.linspace(0.001,1,mesh_size+1)
N_arr = np.linspace(len(det_snr),500,mesh_size+2,dtype=int)

mat = np.zeros((mesh_size,mesh_size+1,mesh_size+2))
for i,mu in enumerate(mu_arr):
    print(i)
    for j,std in enumerate(std_arr):
        for k,N in enumerate(N_arr):
            mat[i,j,k] = total_p(mu,std,N,det_snr)
np.save('data',mat)
#integrate over mu and std
posterior = np.trapz(np.trapz(mat,mu_arr,axis=0),std_arr,axis=0)
plt.plot(N_arr,posterior)
plt.show()
import pdb; pdb.set_trace()

plt.figure()
plt.hist(det_snr,bins=100)
plt.figure()
plt.hist(np.log10(pulse_snrs),bins=100)
plt.show()
