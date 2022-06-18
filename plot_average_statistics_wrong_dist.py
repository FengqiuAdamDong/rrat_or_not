#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import sys
fn = sys.argv[1:]
max_N = []
max_frac = []
max_mu_arr = []
for f in fn:
    data = np.load(f)
    mat = data['data']
    N_arr = data['N']
    mu_arr = data['mu']
    std_arr = data['std']
    snrs = data['snrs']
    dets = data['det']
    frac = data['true_frac']
    p = data['p']
    obs_t = data['obs_t']
    true_mu = data['true_mu']
    mat = mat-np.max(mat)
    mat = np.exp(mat)
    posterior = np.trapz(np.trapz(mat,mu_arr,axis=0),std_arr,axis=0)
    #lets try find the mean, we'll just normalise the likelihood
    posterior = posterior/np.trapz(posterior,N_arr)
    #now find the mean_mu
    # mean = np.trapz(posterior*N_arr,N_arr)
    mN = N_arr[max(posterior)==posterior][0]
    max_N.append(mN/len(snrs))
    det_frac = mN/(obs_t/p)
    max_frac.append(det_frac)
    posterior_mu = np.trapz(np.trapz(mat,std_arr,axis=1),N_arr,axis=1)
    max_mu = mu_arr[posterior_mu == max(posterior_mu)]
    # mean_mu = np.trapz(posterior_mu*mu_arr,mu_arr)
    max_mu_arr.append(max_mu[0])
    #do the same thing for mu


plt.hist(max_N,bins=10)
plt.title("Predicted N/True N")
plt.xlabel("N/N_true")
plt.ylabel("count")
plt.figure()
plt.hist(max_frac,bins=10)
plt.title(f"detected pulse fraction - true fraction:{frac}")
plt.ylabel('count')
plt.xlabel('Pulse fraction')
plt.figure()
plt.hist(max_mu_arr,bins=10)
plt.title(f"predicted mu - true mu:{true_mu}")
plt.xlabel("mu")
plt.ylabel('count')
plt.show()
# plt.plot(N_arr,posterior)
# plt.xlabel('N')
# plt.title(f"# of simulated pulses:{len(snrs)} # of det pulses:{len(dets)}")
# plt.show()
# posterior = np.trapz(np.trapz(mat,std_arr,axis=1),N_arr,axis=1)
# plt.plot(mu_arr,posterior)
# plt.xlabel('mu')
# plt.title(f"# of simulated pulses:{len(snrs)} # of det pulses:{len(dets)}")

# plt.show()
# #marginalise over std
# d_pos = np.trapz(mat,std_arr,axis=1)
# print(d_pos.shape)
# plt.pcolormesh(mu_arr,N_arr,d_pos.T)
# plt.xlabel('mu')
# plt.ylabel('N')
# plt.title(f"# of simulated pulses:{len(snrs)} # of det pulses:{len(dets)}")
# plt.show()

# d_pos = np.trapz(mat,mu_arr,axis=0)
# print(d_pos.shape)
# plt.pcolormesh(std_arr,N_arr,d_pos.T)
# plt.xlabel('sig')
# plt.ylabel('N')
# plt.title(f"# of simulated pulses:{len(snrs)} # of det pulses:{len(dets)}")
# plt.show()

# print(data.__dict__)
