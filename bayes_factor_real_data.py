#!/usr/bin/env python3
import numpy as np
import os
import statistics
import statistics_exp
from matplotlib import pyplot as plt
from scipy import optimize as o
import dill
import warnings
warnings.filterwarnings("ignore")
def plot_mat_exp(mat,N_arr,k_arr,snrs,dets):
    true_k = 0
    max_likelihood_exp = np.max(mat)
    mat = np.exp(mat-np.max(mat))#*np.exp(max_likelihood_exp)
    plt.figure()
    posterior = np.trapz(mat,k_arr,axis=0)
    plt.plot(N_arr,posterior)
    plt.xlabel('N')
    plt.title(f"# of simulated pulses:{len(snrs)} # of det pulses:{len(dets)}")
    plt.figure()
    posterior = np.trapz(mat,N_arr,axis=1)
    plt.plot(k_arr,posterior)
    plt.xlabel('K')
    plt.title(f"# of simulated pulses:{len(snrs)} # of det pulses:{len(dets)}")
    plt.figure()
    #marginalise over std
    plt.pcolormesh(k_arr,N_arr,mat.T)
    plt.xlabel('k')
    plt.ylabel('N')
    plt.title(f"# of simulated pulses:{len(snrs)} # of det pulses:{len(dets)} true k:{true_k}")
    plt.show()

def plot_mat_ln(mat,N_arr,mu_arr,std_arr,snrs,dets,true_mu,true_std):
    max_likelihood_ln = np.max(mat)
    mat = np.exp(mat-np.max(mat))#*np.exp(max_likelihood_ln)
    plt.figure()
    posterior = np.trapz(np.trapz(mat,mu_arr,axis=0),std_arr,axis=0)
    plt.plot(N_arr,posterior)
    plt.xlabel('N')
    plt.title(f"# of simulated pulses:{len(snrs)} # of det pulses:{len(dets)}")
    posterior = np.trapz(np.trapz(mat,mu_arr,axis=0),N_arr,axis=1)
    plt.figure()
    plt.plot(std_arr,posterior)
    plt.xlabel('std')
    plt.title(f"# of simulated pulses:{len(snrs)} # of det pulses:{len(dets)} true mu:{true_mu}")
    posterior = np.trapz(np.trapz(mat,std_arr,axis=1),N_arr,axis=1)
    plt.figure()
    plt.plot(mu_arr,posterior)
    plt.xlabel('mu')
    plt.title(f"# of simulated pulses:{len(snrs)} # of det pulses:{len(dets)} true mu:{true_mu}")
    plt.figure()
    #marginalise over std
    d_pos = np.trapz(mat,std_arr,axis=1)
    print(d_pos.shape)
    plt.pcolormesh(mu_arr,N_arr,d_pos.T)
    plt.xlabel('mu')
    plt.ylabel('N')
    plt.title(f"# of simulated pulses:{len(snrs)} # of det pulses:{len(dets)} true mu:{true_mu}")
    plt.figure()
    d_pos = np.trapz(mat,mu_arr,axis=0)
    print(d_pos.shape)
    plt.pcolormesh(std_arr,N_arr,d_pos.T)
    plt.xlabel('std')
    plt.ylabel('N')
    plt.title(f"# of simulated pulses:{len(snrs)} # of det pulses:{len(dets)} true std:{true_std}")
    plt.show()

if __name__=="__main__":
    import sys
    folder = sys.argv[1]
    odds_ratios = []
    real_det = sys.argv[1]
    obs_t = 166788
    p = 3.025
    with open(real_det,'rb') as inf:
        det_class = dill.load(inf)
    det_snr = []
    for pulse_obj in det_class.sorted_pulses:
        if pulse_obj.det_snr!=-1:
            #-1 means that the snr could not be measured well
            det_snr.append(pulse_obj.det_snr)

    # plt.figure()
    # plt.hist(pulse_snrs,bins=50)
    # plt.figure()
    # plt.hist(det_snr,bins=50)
    det_snr = np.array(det_snr)
    res = o.minimize(statistics.negative_loglike,[0.5,0.1,len(det_snr)],(det_snr),
                method='Nelder-Mead',bounds=[(-2,2),(0.01,2),(len(det_snr),obs_t/p)])
    mu_min = res.x[0]
    std_min = res.x[1]
    N_min = res.x[2]
    print(mu_min,std_min,N_min)
    mesh_size = 80
    exp_mesh_size = 100
    #log normal original distribution
    mu_arr = np.linspace(mu_min-0.5,mu_min+0.5,mesh_size)
    std_arr = np.linspace(std_min*0.5,std_min*2,mesh_size+1)
    N_arr = np.linspace(len(det_snr),N_min*2,mesh_size+2)
    mat = statistics.likelihood_lognorm(mu_arr,std_arr,N_arr,det_snr,mesh_size=mesh_size)
    plot_mat_ln(mat,N_arr,mu_arr,std_arr,det_snr,det_snr,0,0)
    #find the minimum for the exp
    res = o.minimize(statistics_exp.negative_loglike,[1,len(det_snr)],(det_snr),
                method='Nelder-Mead',bounds=[(0,10),(len(det_snr),obs_t/p)])
    # create an array for k
    min_k = res.x[0]
    min_N = res.x[1]
    mu_lim = [min(mu_arr),max(mu_arr)]
    mu_range = (mu_lim[1]-mu_lim[0])/2

    print(min_k,min_N)
    if min_k<mu_range:
        k_lim = [0.01,mu_range*2]
    else:
        k_lim = (min_k-mu_range,min_k+mu_range)
    if 2*min_N > (obs_t/p):
        min_N = obs_t/p/2
    k_arr = np.linspace(k_lim[0],k_lim[1],exp_mesh_size)
    N_arr_exp = np.linspace(len(det_snr),min_N*2,exp_mesh_size*3)
    mat_exp = statistics_exp.likelihood_exp(k_arr,N_arr_exp,det_snr)
    plot_mat_exp(mat_exp,N_arr_exp,k_arr,det_snr,det_snr)
    #lets calculate bayes factor
    range_N = max(N_arr)-min(N_arr)
    range_mu = max(mu_arr)-min(mu_arr)
    range_std = max(std_arr)-min(std_arr)
    #using uniform priors
    max_likelihood_ln = np.max(mat)
    max_likelihood_exp = np.max(mat_exp)
    bayes_numerator = np.log(np.trapz(np.trapz(np.trapz(np.exp(mat-max_likelihood_ln),mu_arr,axis=0),std_arr,axis=0),N_arr,axis=0))+max_likelihood_ln-np.log(1/(range_N*range_mu*range_std))
    bayes_denominator = np.log(np.trapz(np.trapz(np.exp(mat_exp-max_likelihood_exp),k_arr,axis=0),N_arr_exp,axis=0))+max_likelihood_exp-np.log(1/(range_N*range_mu))
    OR = bayes_numerator-bayes_denominator
    # if OR<0:
    #     print(f"OR less than 0 {fn}")
    #     plot_mat_ln(mat,N_arr,mu_arr,std_arr,pulse_snrs,det_snr,true_mu,true_std)
    #     plot_mat_exp(mat_exp,N_arr_exp,k_arr,pulse_snrs,det_snr)


    print('log Odds Ratio in favour of LN model',OR)
