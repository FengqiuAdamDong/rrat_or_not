#!/usr/bin/env python3
import statistics
from scipy.special import gammaln
from multiprocessing import Pool
import os
from simulate_pulse import simulate_pulses_exp
from statistics import n_detect
import numpy as np
from matplotlib import pyplot as plt
def total_p_exp(X):
    k = X['k']
    N = X['N']
    snr_arr = X['snr_arr']
    f = statistics.first_exp(snr_arr,k)
    s = statistics.second_exp(len(snr_arr),k,N)
    n = len(snr_arr)
    log_NCn = gammaln(N+1)-gammaln(n+1)-gammaln(N-n+1)
    # print(log_NCn,f,s)
    return log_NCn+f+s
def negative_loglike(X,det_snr):
    x={"k":X[0],"N":X[1],"snr_arr":det_snr}
    return -1*total_p_exp(x)

def likelihood_exp(k_arr,N_arr,det_snr):
    mat = np.zeros((len(k_arr),len(N_arr)))
    with Pool(50) as po:
        for i,k_i in enumerate(k_arr):
                X = []
                for j,N_i in enumerate(N_arr):
                    X.append({'k':k_i,'N':N_i,'snr_arr':det_snr})
                # for ind,v in enumerate(X):
                    # mat[i,ind] = total_p_exp(v)
                mat[i,:] = po.map(total_p_exp,X)
    return mat
if __name__=='__main__':
    # x = np.linspace(0,5,100)
    # y = p_detect(x)
    # plt.plot(x,y)
    # plt.show()
    pos_array = []
    for a in range(1):
        obs_t = 500000
        k = 1
        p = 2
        frac = 0.5
        pulse_snrs = simulate_pulses_exp(obs_t,p,frac,k)
        det_snr = n_detect(pulse_snrs)
        # plt.figure()
        # plt.hist(pulse_snrs,bins=100)
        # plt.figure()
        # plt.hist(det_snr,bins=100)
        # plt.show()
        # np.save('simulated_pulses_0.65_0.1',[pulse_snrs,det_snr])
        # pulses = np.load('simulated_pulses_0.65_0.1.npy',allow_pickle=1)
        # pulse_snrs = pulses[0]
        # det_snr = pulses[1]
        print("number of generated pulses",len(pulse_snrs),"number of detections",len(det_snr))
        mesh_size = 50
        # # create a mesh grid of N, mu and stds
        k_arr = np.linspace(k-0.4,k+0.5,mesh_size)
        N_arr = np.linspace((obs_t*frac/p)*0.5,(obs_t*frac/p)*2,mesh_size+1,dtype=int)
        # N_arr = np.linspace(len(det_snr),obs_t/p,mesh_size+1,dtype=int)
        mat = likelihood_exp(k_arr,N_arr,det_snr)
        fn = f"d_{a}"
        print('saving',fn)
        save_dir = f"obs_{obs_t}_k_{k}_p_{p}_frac_{frac}"
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        fn = f"{save_dir}/{fn}"
        np.savez(fn,data=mat,k=k_arr,N=N_arr,snrs=pulse_snrs,
                    det=det_snr,true_k=k,p=p,true_frac=frac,obs_t=obs_t)


        # import pdb; pdb.set_trace()
        mat = mat-np.max(mat)
        mat = np.exp(mat)
        # integrate over mu and std
        posterior = np.trapz(mat,k_arr,axis=0)
        pos_array.append(posterior)
        plt.plot(N_arr,posterior)
        plt.figure()
        plt.pcolormesh(k_arr,N_arr,mat.T)
        # plt.colorbar()
        plt.show()
