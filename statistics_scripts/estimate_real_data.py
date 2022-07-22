#!/usr/bin/env python3
#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from simulate_pulse import simulate_pulses
from scipy.stats import norm
from math import comb
from scipy.special import gammaln
from multiprocessing import Pool
def p_detect(snr,decay_rate,lower_cutoff=5.98):
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
    p_det = np.log(p_detect(snr,1))
    snr_p = np.log(snr_distribution(snr,mu,std))

    return np.sum(p_det+snr_p)

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

def total_p(X):
    mu = X['mu']
    std = X['std']
    N = X['N']
    snr_arr = X['snr_arr']
    f = first(snr_arr,mu,std)
    s = second(len(snr_arr),mu,std,N)
    # NCn = comb(N,len(snr_arr))
    # print(NCn,f,s)
    # NCn = 1
    n = len(snr_arr)
    log_NCn = gammaln(N+1)-gammaln(n+1)-gammaln(N-n+1)
    # print(log_NCn,f,s)
    return log_NCn+f+s

snrs = np.linspace(0,10,1000)
plt.plot(snrs,p_detect(snrs,1))
plt.title('Detection curve')
plt.xlabel('snr')
plt.ylabel('p_det')

plt.show()
obs_t = 172486.8
mu = 0.85
std = 0.16
p = 3.02535
det_snr = np.load('snr_arr_J2355.npy')
g_6 =[]
for d in det_snr:
    if d>=6:
        g_6.append(d)
det_snr = np.array(g_6)
print(np.min(det_snr))

mesh_size = 50
# # create a mesh grid of N, mu and stds
mu_arr = np.linspace(mu-0.15,mu+0.1,mesh_size)
std_arr = np.linspace(std-0.1,std+0.1,mesh_size+1)
N_arr = np.linspace(len(det_snr),2000,mesh_size+2,dtype=int)
mat = np.zeros((mesh_size,mesh_size+1,mesh_size+2))
with Pool(25) as p:
    for i,mu in enumerate(mu_arr):
        print(i)
        for j,std in enumerate(std_arr):
            X = []
            for k,N in enumerate(N_arr):
                X.append({'mu':mu,'std':std,'N':N,'snr_arr':det_snr})
            mat[i,j,:] = p.map(total_p,X)


fn = "d_2355"
print('saving',fn)
np.savez(fn,data=mat,mu=mu_arr,std=std_arr,N=N_arr,det=det_snr)
mat = mat-np.max(mat)
mat = np.exp(mat)
# integrate over mu and std
posterior = np.trapz(np.trapz(mat,mu_arr,axis=0),std_arr,axis=0)
plt.plot(N_arr,posterior)
plt.show()
