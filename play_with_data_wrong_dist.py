#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import sys
data = np.load(sys.argv[1])

mat = data['data']
N_arr = data['N']
mu_arr = data['mu']
std_arr = data['std']
snrs = data['snrs']
dets = data['det']
true_mu = data['true_k']
true_std = data['true_scale']
mat = mat-np.max(mat)
mat = np.exp(mat)

posterior = np.trapz(np.trapz(mat,mu_arr,axis=0),std_arr,axis=0)
plt.plot(N_arr,posterior)
plt.xlabel('N')
plt.title(f"# of simulated pulses:{len(snrs)} # of det pulses:{len(dets)}")
plt.show()
posterior = np.trapz(np.trapz(mat,std_arr,axis=1),N_arr,axis=1)
plt.plot(mu_arr,posterior)
plt.xlabel('mu')
plt.title(f"# of simulated pulses:{len(snrs)} # of det pulses:{len(dets)} true mu:{true_mu}")

plt.show()
#marginalise over std
d_pos = np.trapz(mat,std_arr,axis=1)
print(d_pos.shape)
plt.pcolormesh(mu_arr,N_arr,d_pos.T)
plt.xlabel('mu')
plt.ylabel('N')
plt.title(f"# of simulated pulses:{len(snrs)} # of det pulses:{len(dets)} true mu:{true_mu}")
plt.show()

d_pos = np.trapz(mat,mu_arr,axis=0)
print(d_pos.shape)
plt.pcolormesh(std_arr,N_arr,d_pos.T)
plt.xlabel('std')
plt.ylabel('N')
plt.title(f"# of simulated pulses:{len(snrs)} # of det pulses:{len(dets)} true std:{true_std}")
plt.show()

print(data.__dict__)
