#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from math import comb
from scipy.special import gammaln
from multiprocessing import Pool
import os
import dill
import sys
def p_detect(snr,decay_rate=2,k=5.5762911,x0=2.12284101,L=1,lower_cutoff=1.5):
    #this will just be an exponential rise at some center
    #added a decay rate variable just so things are compatible
    p = L/(1+np.exp(-k*(snr-x0)))
    p[snr<=lower_cutoff]=0
    return p

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
    real_det = sys.argv[1]
    with open(real_det,'rb') as inf:
        det_class = dill.load(inf)
    det_snr = []
    for pulse_obj in det_class.sorted_pulses:
        if pulse_obj.det_snr!=-1:
            #-1 means that the snr could not be measured well
            det_snr.append(pulse_obj.det_snr)
    det_snr = np.array(det_snr)
    x = np.linspace(0.5,4)
    y = p_detect(x)

    n,bins,patches = plt.hist(det_snr,bins=20)
    plt.title(real_det)
    plt.xlabel('snr')
    plt.ylabel('count')
    plt.figure()

    plt.plot(x,y)
    plt.xlabel('snr')
    plt.ylabel('p_detect')
    bins = bins+np.diff(bins)[0]/2
    adjust_val = p_detect(bins)
    adjusted_n = n/(adjust_val[:-1])
    adjusted_sum = np.sum(adjusted_n)
    print(adjusted_sum)
    plt.figure()
    plt.plot(bins[:-1],n)
    plt.ylabel('adjusted_n')
    plt.xlabel('snr')
    plt.show()

    #these are guesses of the parameter
    mu = np.mean(np.log10(det_snr))
    std = np.std(np.log10(det_snr))
    N=len(det_snr)
    print(mu,std)
    if 1:
        mesh_size = 20
        # # create a mesh grid of N, mu and stds
        mu_arr = np.linspace(-mu,mu*1.5,mesh_size)
        std_arr = np.linspace(0.15,std*3,mesh_size+1)
        N_arr = np.linspace(N,N*100,mesh_size+2,dtype=int)
        mat = np.zeros((mesh_size,mesh_size+1,mesh_size+2))
        with Pool(8) as po:
            for i,mu_i in enumerate(mu_arr):
                for j,std_i in enumerate(std_arr):
                    X = []
                    for k,N_i in enumerate(N_arr):
                        X.append({'mu':mu_i,'std':std_i,'N':N_i,'snr_arr':det_snr})
                    mat[i,j,:] = po.map(total_p,X)

        fn = f"{real_det}"
        print('saving',fn)
        save_dir = f"{real_det}_fol"
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        fn = f"{save_dir}/{fn}"
        np.savez(fn,data=mat,mu=mu_arr,std=std_arr,N=N_arr,det=det_snr,adjusted_sum=adjusted_sum)
        mat = mat-np.max(mat)
        mat = np.exp(mat)
        # integrate over mu and std
        posterior = np.trapz(np.trapz(mat,mu_arr,axis=0),std_arr,axis=0)

    plt.plot(N_arr,posterior)
    plt.xlabel('N')
    plt.title(f"# of det pulses:{len(det_snr)}")
    plt.show()

    plt.figure()
    plt.hist(det_snr,bins=100)
    plt.xlabel('detected snr')
    plt.show()
