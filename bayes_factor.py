#!/usr/bin/env python3
import numpy as np
import statistics
import statistics_exp
from matplotlib import pyplot as plt

def load_data(fn):
    #load files and det snrs
    data = np.load(fn)
    #try load log norm
    try:
        mat = data['data']
        N_arr = data['N']
        mu_arr = data['mu']
        std_arr = data['std']
        snrs = data['snrs']
        dets = data['det']
        true_mu = data['true_mu']
        true_std = data['true_std']
        params = {'N':N_arr,'mu':mu_arr,'std':std_arr}
    except:
        #it's exponential
        mat = data['data']
        N_arr = data['N']
        k_arr = data['k']
        snrs = data['snrs']
        dets = data['det']
        true_k = data['true_k']
        params = {'N':N_arr,'k':k_arr}

    p  = data['p']
    obs_t = data['obs_t']
    return snrs,dets,params,mat,p,obs_t

if __name__=="__main__":
    import sys
    fn = sys.argv[1]
    pulse_snrs,det_snr,params,mat,p,obs_t = load_data(fn)
    if len(params.keys())==3:
        Orig_dist_ln = True
    else:
        Orig_dist_ln = False

    if Orig_dist_ln:
        #log normal original distribution
        N_arr = params['N']
        mu_arr = params['mu']
        std_arr = params['std']
        mesh_size = len(mu_arr)
        # mat = statistics.likelihood_lognorm(mu_arr,std_arr,N_arr,det_snr,mesh_size=mesh_size)
        max_mat = np.exp(mat-np.max(mat))
        max_likelihood_ln = np.max(mat)
        posterior = np.log(np.trapz(np.trapz(max_mat,mu_arr,axis=0),std_arr,axis=0))+max_likelihood_ln
        # create an array for k
        exp_mesh_size = 100
        k = 2.95
        k_range = max(mu_arr)-min(mu_arr)
        k_arr = np.linspace(k-k_range/2,k+k_range/2,exp_mesh_size)
        N_arr_exp = np.linspace(len(det_snr),obs_t/p,exp_mesh_size+1)
        mat_exp = statistics_exp.likelihood_exp(k_arr,N_arr_exp,det_snr,exp_mesh_size)
        max_likelihood_exp =  np.max(mat_exp)
        mat_exp = np.exp(mat_exp - max_likelihood_exp)
        # mat_exp = np.exp(mat_exp)
        # posterior_exp = np.trapz(mat_exp,k_arr,axis=0)
        # max_likelihood = np.log(1)
        #integrate
        posterior_exp = np.log(np.trapz(mat_exp,k_arr,axis=0)) +max_likelihood_exp
        plt.figure()
        plt.plot(N_arr_exp,posterior_exp)
        plt.show()
        plt.figure()
        plt.pcolormesh(k_arr,N_arr_exp,mat_exp.T)
        plt.colorbar()
        plt.figure()
        plt.plot(N_arr,posterior)
        plt.xlabel('N')
        plt.title(f"# of simulated pulses:{len(pulse_snrs)} # of det pulses:{len(det_snr)}")
        plt.show()
        #lets calculate bayes factor
        range_N = max(N_arr)-min(N_arr)
        range_mu = max(mu_arr)-min(mu_arr)
        range_std = max(std_arr)-min(std_arr)
        #using uniform priors
        bayes_numerator = np.log(np.trapz(np.trapz(np.trapz(max_mat,mu_arr,axis=0),std_arr,axis=0),N_arr,axis=0))+max_likelihood_ln-np.log(1/(range_N*range_mu*range_std))
        bayes_denominator = np.log(np.trapz(np.trapz(mat_exp,k_arr,axis=0),N_arr_exp,axis=0))+max_likelihood_exp-np.log(1/(range_N*range_mu))
        print('log Odds Ratio in favour of LN model',bayes_numerator-bayes_denominator)
        import pdb; pdb.set_trace()
    else:
        #log normal original distribution
        N_arr = params['N']
        k_arr = params['k']
        mesh_size = len(k_arr)
        # mat = statistics.likelihood_lognorm(mu_arr,std_arr,N_arr,det_snr,mesh_size=mesh_size)
        max_mat = np.exp(mat-np.max(mat))
        max_likelihood_exp = np.max(mat)
        posterior_exp = np.log(np.trapz(max_mat,k_arr,axis=0))+max_likelihood_exp

        # create an array for k
        ln_mesh_size = 20
        mu = 0.5
        std = 0.1
        mu_range = max(k_arr)-min(k_arr)
        #define parameters
        mu_arr = np.linspace(mu-mu_range/2,mu+4*mu_range/2,ln_mesh_size)
        std_arr = np.linspace(std-std/2,std+std/2,ln_mesh_size+1)
        N_arr_ln = np.linspace(len(det_snr),obs_t/p,ln_mesh_size+2)
        #push them into likelihoods
        mat_ln = statistics.likelihood_lognorm(mu_arr,std_arr,N_arr_ln,det_snr,ln_mesh_size)
        max_likelihood_ln =  np.max(mat_ln)
        mat_ln = np.exp(mat_ln - max_likelihood_ln)
        #integrate
        posterior_ln = np.log(np.trapz(np.trapz(mat_ln,mu_arr,axis=0),std_arr,axis=0))+max_likelihood_ln
        plt.figure()
        plt.plot(N_arr,np.exp(posterior_exp))
        plt.show()
        plt.figure()
        plt.pcolormesh(k_arr,N_arr,np.exp(mat).T)
        plt.colorbar()
        plt.figure()
        plt.plot(N_arr_ln,posterior_ln)
        plt.xlabel('N')
        plt.title(f"# of simulated pulses:{len(pulse_snrs)} # of det pulses:{len(det_snr)}")
        plt.show()
        #lets calculate bayes factor
        range_N = max(N_arr)-min(N_arr)
        range_mu = max(mu_arr)-min(mu_arr)
        range_std = max(std_arr)-min(std_arr)
        #using uniform priors
        bayes_numerator = np.log(np.trapz(np.trapz(np.trapz(max_mat,mu_arr,axis=0),std_arr,axis=0),N_arr,axis=0))+max_likelihood_ln-np.log(1/(range_N*range_mu*range_std))
        bayes_denominator = np.log(np.trapz(np.trapz(mat_exp,k_arr,axis=0),N_arr_exp,axis=0))+max_likelihood_exp-np.log(1/(range_N*range_mu))
        print('log Odds Ratio in favour of LN model',bayes_numerator-bayes_denominator)
        import pdb; pdb.set_trace()
