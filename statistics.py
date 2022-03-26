#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from simulate_pulse import simulate_pulses
from scipy.stats import norm
from math import comb
from scipy.special import gammaln
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
    pdf = norm.pdf(np.log10(snr),loc=mu,scale=std)
    #convert to pdf for SNR and not log SNR
    pdf = pdf/(snr*np.log(10))
    return pdf

def first(snr,mu,std):
    p_det = p_detect(snr,2)
    snr_p = snr_distribution(snr,mu,std)
    return np.sum(np.log(p_det*snr_p))

def second(n,mu,std,N):
    snr_arr = np.linspace(0.1,100,10000)
    p_snr = snr_distribution(snr_arr,mu,std)
    p_not_det = 1-(p_detect(snr_arr,2))
    p_second = p_snr*p_not_det
    #integrate over flux
    # plt.plot(snr_arr,p_snr)
    # plt.show()
    p_second_int = np.log(np.trapz(p_second,snr_arr))
    if p_second_int>1:
        # print(p_second_int)
        p_second_int=1
    return p_second_int*(N-n)

def total_p(mu,std,N,snr_arr):
    f = first(snr_arr,mu,std)
    s = second(len(snr_arr),mu,std,N)
    # NCn = comb(N,len(snr_arr))
    # print(NCn,f,s)
    # NCn = 1
    n = len(snr_arr)
    log_NCn = gammaln(N+1)-gammaln(n+1)-gammaln(N-n+1)
    # print(log_NCn,f,s)
    return log_NCn+f+s

obs_t =180000
mu = 0.63
std = 0.1
p = 2
frac = 0.5
pulse_snrs = simulate_pulses(obs_t,p,frac,mu,std)
det_snr = n_detect(pulse_snrs)
# np.save('simulated_pulses_0.65_0.1',[pulse_snrs,det_snr])
# pulses = np.load('simulated_pulses_0.65_0.1.npy',allow_pickle=1)
# pulse_snrs = pulses[0]
# det_snr = pulses[1]
print(len(pulse_snrs),len(det_snr))
mesh_size = 100
# # create a mesh grid of N, mu and stds
mu_arr = np.linspace(mu-0.2,mu+0.2,mesh_size)
std_arr = np.linspace(std-0.05,std+0.05,mesh_size+1)
N_arr = np.linspace(len(det_snr),(obs_t/p)*0.5,mesh_size+2,dtype=int)
mat = np.zeros((mesh_size,mesh_size+1,mesh_size+2))
for i,mu in enumerate(mu_arr):
    print(i)
    for j,std in enumerate(std_arr):
        for k,N in enumerate(N_arr):
            mat[i,j,k] = total_p(mu,std,N,det_snr)
np.savez('data',data=mat,mu=mu_arr,std=std_arr,N=N_arr,snrs=pulse_snrs,det=det_snr)
mat = mat-np.max(mat)
mat = np.exp(mat)
# integrate over mu and std
posterior = np.trapz(np.trapz(mat,mu_arr,axis=0),std_arr,axis=0)
# create a mesh grid of N, mu and stds
# mesh_size=10000
# mu = 0.8
# std = 0.1
# N_arr = np.linspace(len(det_snr),2500,mesh_size+2,dtype=int)
# mat = np.zeros(mesh_size+2)
# for k,N in enumerate(N_arr):
#     mat[k] = total_p(mu,std,N,det_snr)
#     # print(mat[k])
# mat = mat-np.max(mat)
# mat = np.exp(mat)
# posterior = mat


plt.plot(N_arr,posterior)
plt.xlabel('N')
plt.title(f"# of simulated pulses:{len(pulse_snrs)} # of det pulses:{len(det_snr)}")
plt.show()
import pdb; pdb.set_trace()

plt.figure()
plt.hist(det_snr,bins=100)
plt.figure()
plt.hist(np.log10(pulse_snrs),bins=100)
plt.show()
