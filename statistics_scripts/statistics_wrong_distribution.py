#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from simulate_pulse import simulate_pulses
from simulate_pulse import simulate_pulses_pow
from scipy.stats import norm
from math import comb
from scipy.special import gammaln
from multiprocessing import Pool
import os
def p_detect_0(snr,decay_rate,lower_cutoff=6):
    #this will just be an exponential rise at some center
    p = 1-np.exp(-1*decay_rate*(snr-lower_cutoff))
    p[snr<lower_cutoff] = 0
    return p

def p_detect(snr,decay_rate=2,k=5.5762911,x0=2.12284101,L=1):
    #this will just be an exponential rise at some center
    #added a decay rate variable just so things are compatible
    return L/(1+np.exp(-k*(snr-x0)))

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


if __name__=='__main__':
    # x = np.linspace(0,5,100)
    # y = p_detect(x)
    # plt.plot(x,y)
    # plt.show()
    pos_array = []
    for a in range(1):
        obs_t =500000
        mu = 0.3
        std = 0.05
        p = 2
        frac = 1
        #power law parameters
        _k = 4
        scale = 6
        lower_cutoff = 0.5
        # pulse_snrs = simulate_pulses(obs_t,p,frac,mu,std)
        pulse_snrs = simulate_pulses_pow(obs_t,p,frac,
                                         k=_k,scale=scale,lower_cutoff=lower_cutoff)

        det_snr = n_detect(pulse_snrs)
        print("number of generated pulses",len(pulse_snrs),"number of detections",len(det_snr))
        if 1:
            mesh_size = 50
            # # create a mesh grid of N, mu and stds
            mu_arr = np.linspace(0.32,0.35,mesh_size)
            std_arr = np.linspace(0.162,0.175,mesh_size+1)
            # N_arr = np.linspace(100000,265000,mesh_size+2,dtype=int)

            N_arr = np.linspace(135000,145000,mesh_size+2,dtype=int)
            mat = np.zeros((mesh_size,mesh_size+1,mesh_size+2))
            with Pool(8) as po:
                for i,mu_i in enumerate(mu_arr):
                    for j,std_i in enumerate(std_arr):
                        X = []
                        for k,N_i in enumerate(N_arr):
                            X.append({'mu':mu_i,'std':std_i,'N':N_i,'snr_arr':det_snr})
                        mat[i,j,:] = po.map(total_p,X)

            fn = f"d_{a}"
            print('saving',fn)
            save_dir = f"obs_{obs_t}_k_{_k}_s_{scale}_cut_{lower_cutoff}_p_{p}_frac_{frac}"
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            fn = f"{save_dir}/{fn}"
            np.savez(fn,data=mat,mu=mu_arr,std=std_arr,N=N_arr,snrs=pulse_snrs,
                     det=det_snr,true_k=k,true_scale=scale,lower_cutoff=lower_cutoff,
                     p=p,true_frac=frac,obs_t=obs_t)
            mat = mat-np.max(mat)
            mat = np.exp(mat)
            # integrate over mu and std
            posterior = np.trapz(np.trapz(mat,mu_arr,axis=0),std_arr,axis=0)
            pos_array.append(posterior)
        else:
            # create a mesh grid of N, mu and stds
            mesh_size=10000
            mu = 0.63
            std = 0.1
            N_arr = np.linspace(len(det_snr),(obs_t/p),mesh_size+2,dtype=int)
            mat = np.zeros(mesh_size+2)
            for k,N in enumerate(N_arr):
                mat[k] = total_p({'mu':mu,'std':std,'N':N,'snr_arr':det_snr})
                # print(mat[k])
            mat = mat-np.max(mat)
            mat = np.exp(mat)
            posterior = mat

    # np.save('posteriors',pos_array)

    plt.plot(N_arr,posterior)
    plt.xlabel('N')
    plt.title(f"# of simulated pulses:{len(pulse_snrs)} # of det pulses:{len(det_snr)}")
    plt.show()

    plt.figure()
    plt.hist(det_snr,bins=100)
    plt.xlabel('detected snr')
    plt.figure()
    plt.hist(pulse_snrs,bins=100)
    plt.xlabel("emmitted snr")
    plt.show()
