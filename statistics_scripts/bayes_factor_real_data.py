#!/usr/bin/env python3
#paths
import sys
sys.path
sys.path.append("/home/adam/Documents/rrat_or_not/injection_scripts/")
print(sys.path)

import numpy as np
import os
import statistics
import statistics_exp
from matplotlib import pyplot as plt
from scipy import optimize as o
import dill
import warnings
import inject_stats



warnings.filterwarnings("ignore")
def N_to_pfrac(x):
    obs_t=1088
    p=1.235
    total = obs_t/p
    return (x/total)
def pfrac_to_N(x):
    obs_t=1088
    p=1.235
    total = obs_t/p
    return (total*X)

def plot_mat_exp(mat,N_arr,k_arr,snrs,dets):
    true_k = 0
    max_likelihood_exp = np.max(mat)
    mat = np.exp(mat-np.max(mat))#*np.exp(max_likelihood_exp)
    fig1,ax1 = plt.subplots()
    posterior = np.trapz(mat,k_arr,axis=0)
    ax1.plot(N_arr,posterior)
    secax1 = ax1.secondary_xaxis('top',functions = (N_to_pfrac,N_to_pfrac))
    secax1.set_xlabel('Pulsing Fraction')
    ax1.set_xlabel('N')
    ax1.set_title(f"# of det pulses:{len(dets)}")



    plt.figure()
    posterior = np.trapz(mat,N_arr,axis=1)
    plt.plot(k_arr,posterior)
    plt.xlabel('K')
    plt.title(f"# of det pulses:{len(dets)}")



    fig3,ax3 = plt.subplots()
    #marginalise over std
    plt.pcolormesh(k_arr,N_arr,mat.T)
    secax3 = ax3.secondary_yaxis('right',functions = (N_to_pfrac,N_to_pfrac))
    secax3.set_ylabel('Pulsing Fraction')
    ax3.set_xlabel('k')
    ax3.set_ylabel('N')
    ax3.set_title(f"# of det pulses:{len(dets)}")
    plt.show()

def plot_mat_ln(mat,N_arr,mu_arr,std_arr,snrs,dets,true_mu,true_std):
    max_likelihood_ln = np.max(mat)
    mat = np.exp(mat-np.max(mat))#*np.exp(max_likelihood_ln)
    fig1,ax1 = plt.subplots()
    posterior = np.trapz(np.trapz(mat,mu_arr,axis=0),std_arr,axis=0)
    ax1.plot(N_arr,posterior)
    secax1 = ax1.secondary_xaxis('top',functions = (N_to_pfrac,N_to_pfrac))
    secax1.set_xlabel('Pulsing Fraction')
    ax1.set_xlabel('N')
    ax1.set_title(f"# of det pulses:{len(dets)}")
    #set a secondary axis
    posterior = np.trapz(np.trapz(mat,mu_arr,axis=0),N_arr,axis=1)
    plt.figure()
    plt.plot(std_arr,posterior)
    plt.xlabel('std')
    plt.title(f"# of det pulses:{len(dets)}")
    posterior = np.trapz(np.trapz(mat,std_arr,axis=1),N_arr,axis=1)
    plt.figure()
    plt.plot(mu_arr,posterior)
    plt.xlabel('mu')
    plt.title(f"# of det pulses:{len(dets)}")
    fig2,ax2 = plt.subplots()
    #marginalise over std
    d_pos = np.trapz(mat,std_arr,axis=1)
    print(d_pos.shape)
    ax2.pcolormesh(mu_arr,N_arr,d_pos.T)
    ax2.set_xlabel('mu')
    ax2.set_ylabel('N')
    secax2 = ax2.secondary_yaxis('right',functions = (N_to_pfrac,N_to_pfrac))
    secax2.set_ylabel('Pulsing Fraction')
    ax2.set_title(f"# of det pulses:{len(dets)}")
    fig3,ax3 = plt.subplots()
    d_pos = np.trapz(mat,mu_arr,axis=0)
    print(d_pos.shape)
    ax3.pcolormesh(std_arr,N_arr,d_pos.T)
    secax3 = ax3.secondary_yaxis('right',functions = (N_to_pfrac,N_to_pfrac))
    secax3.set_ylabel('Pulsing Fraction')
    plt.xlabel('std')
    plt.ylabel('N')
    plt.title(f"# of det pulses:{len(dets)}")
    plt.show()

if __name__=="__main__":
    import sys
    odds_ratios = []
    real_det = sys.argv[1]
    # obs_t = 1088
    obs_t = 1088.84
    p = 1.235
    with open(real_det,'rb') as inf:
        det_class = dill.load(inf)
    det_snr = []
    for pulse_obj in det_class.sorted_pulses:
        if pulse_obj.det_snr!=-1:
            #-1 means that the snr could not be measured well
            det_snr.append(pulse_obj.det_snr)
    #lets filter the det_SNRs too
    det_snr = np.array(det_snr)
    det_snr = det_snr[det_snr>2.3]
    snr_thresh = 1
    det_snr = np.array(det_snr)
    det_snr = det_snr[det_snr>snr_thresh]
    plt.title('Histogram of detected pulses')
    plt.hist(det_snr,bins=50,density=True)
    plt.xlabel('SNR')
    plt.ylabel('count')
    det_snr = np.array(det_snr)
    res = o.minimize(statistics.negative_loglike,[0.5,0.1,len(det_snr)],(det_snr),
                method='Nelder-Mead',bounds=[(-2,2),(0.01,2),(len(det_snr),4*obs_t/p)])
    mu_min = res.x[0]
    std_min = res.x[1]
    N_min = res.x[2]
    snr_s = np.linspace(1,10,1000)
    y = statistics.snr_distribution(snr_s,mu_min,std_min)
    plt.plot(snr_s,y)
    plt.show()

    print(mu_min,std_min,N_min)
    mesh_size = 50
    exp_mesh_size = 150
    #log normal original distribution
    mu_arr = np.linspace(mu_min-0.5,mu_min+0.5,mesh_size)
    std_arr = np.linspace(std_min*0.5,std_min*2,mesh_size+1)
    N_arr = np.linspace(len(det_snr),3*obs_t/p,mesh_size+2)
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
    N_arr_exp = np.linspace(len(det_snr),obs_t/p,exp_mesh_size*3)
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
