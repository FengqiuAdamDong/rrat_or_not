#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
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

def p_detect(snr):
    #this will just be an exponential rise at some center
    #added a decay rate variable just so things are compatible
    #load inj statistics
    popt = np.load('det_fun_params.npy',allow_pickle=1)
    k = popt[0]
    x0 = popt[1]
    # print(k,x0)
    L = 1
    # x = np.linspace(1,6)
    # detect_plot =  L/(1+np.exp(-k*(x-x0)))
    # plt.plot(x,detect_plot)
    # plt.show()
    # x0=x0+5
    detection_fn = L/(1+np.exp(-k*(snr-x0)))

    return detection_fn
def n_detect(snr_emit):
    #snr emit is the snr that the emitted pulse has
    p = p_detect(snr_emit)
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

def log_snr_distribution_exp(snr,k):
    #create meshgrids
    logpdf = np.log(k)-k*snr
    return logpdf

def first_exp(snr,k):
    p_det = np.log(p_detect(snr))
    snr_p = log_snr_distribution_exp(snr,k)
    return np.sum(p_det+snr_p)

def second_exp(n,k,N):
    snr_arr = np.linspace(-20,2,100000)
    snr_arr = np.exp(snr_arr)
    p_snr = log_snr_distribution_exp(snr_arr,k)
    p_not_det = np.log(1-(p_detect(snr_arr)))
    p_second = p_snr+p_not_det
    #integrate over flux
    # plt.plot(snr_arr,p_second)
    # plt.title(f"{k}")
    # plt.show()
    #use the log sum exp trick when integrating to prevent overflows
    p_second_int = np.log(np.trapz(np.exp(p_second-np.max(p_second)),snr_arr))+np.max(p_second)
    if p_second_int>1:
        print(p_second_int)
        p_second_int=1
    return p_second_int*(N-n)

def first(snr,mu,std):
    p_det = p_detect(snr)
    snr_p = snr_distribution(snr,mu,std)
    return np.sum(np.log(p_det*snr_p))

def second(n,mu,std,N):
    #get a logspace
    snr_arr = np.linspace(0.01,100,10000)
    #take log of the snr distribution
    p_snr = np.log(snr_distribution(snr_arr,mu,std))
    p_not_det = np.log(1-(p_detect(snr_arr)))
    #combine the two terms
    p_second = p_snr+p_not_det
    #integrate over flux
    # plt.figure()
    # plt.plot(snr_arr,p_second)
    # plt.show()
    # exponentiate and integrate, re log using the log sum exp trick to prevent overflows (and low numbers)
    p_second_int = np.log(np.trapz(np.exp(p_second-np.max(p_second)),snr_arr))+np.max(p_second)
    if p_second_int>1:
        print(p_second_int)
        p_second_int=1
    return p_second_int*(N-n)

def total_p(X):
    mu = X['mu']
    std = X['std']
    N = X['N']
    snr_arr = X['snr_arr']
    # if (mu > 0.095) & (mu<1.1):
    #     if (std > 0.095) & (std<1.1) :
    #         if (N > 24000) & (N < 26000):
    #             import pdb; pdb.set_trace()
    f = first(snr_arr,mu,std)
    s = second(len(snr_arr),mu,std,N)
    # NCn = comb(N,len(snr_arr))
    # print(NCn,f,s)
    # NCn = 1
    n = len(snr_arr)
    log_NCn = gammaln(N+1)-gammaln(n+1)-gammaln(N-n+1)
    # import pdb; pdb.set_trace()
    # print(log_NCn,f,s)
    return log_NCn+f+s

def negative_loglike(X,det_snr):
    x={"mu":X[0],"std":X[1],"N":X[2],"snr_arr":det_snr}
    return -1*total_p(x)

def likelihood_lognorm(mu_arr,std_arr,N_arr,det_snr,mesh_size=20):
    # # create a mesh grid of N, mu and stds
    mat = np.zeros((mesh_size,mesh_size+1,mesh_size+2))
    with Pool(50) as po:
        for i,mu_i in enumerate(mu_arr):
            for j,std_i in enumerate(std_arr):
                X = []
                for k,N_i in enumerate(N_arr):
                    X.append({'mu':mu_i,'std':std_i,'N':N_i,'snr_arr':det_snr})
                mat[i,j,:] = po.map(total_p,X)
                # for ind,v in enumerate(X):
                    # mat[i,j,ind] = total_p(v)
    return mat




if __name__=='__main__':
    from simulate_pulse import simulate_pulses
    from simulate_pulse import simulate_pulses_exp

    # x = np.linspace(0,5,100)
    # y = p_detect(x)
    # plt.plot(x,y)
    # plt.show()
    pos_array = []
    for a in range(1):
        obs_t =500000
        mu = 0.3
        std = 0.1
        p = 2
        frac = 0.001
        pulse_snrs = simulate_pulses(obs_t,p,frac,mu,std)
        mesh_size = 50

        det_snr = n_detect(pulse_snrs)
        mu_arr = np.linspace(mu-0.1,mu+0.05,mesh_size)
        std_arr = np.linspace(std-0.04,std+0.05,mesh_size+1)
        N_arr = np.linspace((obs_t*frac/p)*0.5,(obs_t*frac/p)*2,mesh_size+2,dtype=int)
        print("number of generated pulses",len(pulse_snrs),"number of detections",len(det_snr))

        # N_arr = np.linspace(len(det_snr),(obs_t/p)*frac*2,mesh_size+2,dtype=int)

        mat = likelihood_lognorm(mu_arr,std_arr,N_arr,det_snr,mesh_size)
        fn = f"d_{a}"
        print('saving',fn)
        save_dir = f"obs_{obs_t}_mu_{mu}_std_{std}_p_{p}_frac_{frac}"
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        fn = f"{save_dir}/{fn}"
        np.savez(fn,data=mat,mu=mu_arr,std=std_arr,N=N_arr,snrs=pulse_snrs,
                    det=det_snr,true_mu=mu,true_std=std,p=p,true_frac=frac,obs_t=obs_t)
        # import pdb; pdb.set_trace()
        # mat = mat-np.max(mat)
        mat = np.exp(mat)
        # integrate over mu and std
        posterior = np.trapz(np.trapz(mat,mu_arr,axis=0),std_arr,axis=0)
        pos_array.append(posterior)

        # np.save('simulated_pulses_0.65_0.1',[pulse_snrs,det_snr])
        # pulses = np.load('simulated_pulses_0.65_0.1.npy',allow_pickle=1)
        # pulse_snrs = pulses[0]
        # det_snr = pulses[1]

    np.save('posteriors',pos_array)
    plt.figure()
    plt.plot(N_arr,posterior)
    plt.xlabel('N')
    plt.title(f"# of simulated pulses:{len(pulse_snrs)} # of det pulses:{len(det_snr)}")
    plt.show()

    plt.figure()
    plt.hist(det_snr,bins=100)
    plt.title(f"total number of pulses:{len(det_snr)}")
    plt.xlabel(f"detected snr")
    plt.figure()
    plt.hist(pulse_snrs,bins=100)
    plt.xlabel("emmitted snr")
    plt.title(f"total number of pulses:{len(pulse_snrs)}")
    plt.show()
