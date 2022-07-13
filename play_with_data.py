#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import sys
plt.rcParams['font.size']= 16
data = np.load(sys.argv[1])

mat = data['data']
N_arr = data['N']
mu_arr = data['mu']
std_arr = data['std']
snrs = data['snrs']
dets = data['det']
true_mu = data['true_mu']
true_std = data['true_std']
p = data['p']
f = data['true_frac']
O = data['obs_t']
mat = mat-np.max(mat)
mat = np.exp(mat)
plt.figure()
plt.hist(snrs,bins=50)
plt.title(f"total number of pulses:{len(snrs)}")
plt.xlabel("emitted snr")
plt.figure()
plt.hist(dets,bins=50)
plt.title(f"total number of pulses:{len(dets)}")
plt.xlabel(f"detected snr")
plt.show()
posterior = np.trapz(np.trapz(mat,mu_arr,axis=0),std_arr,axis=0)
plt.plot(N_arr,posterior)
plt.xlabel(r'$N_e$')
plt.title(f"# of simulated pulses:{len(snrs)} # of det pulses:{len(dets)}")
plt.show()
posterior = np.trapz(np.trapz(mat,std_arr,axis=1),N_arr,axis=1)
plt.plot(mu_arr,posterior)
plt.xlabel(r'$\mu$')
plt.title(f"# of simulated pulses:{len(snrs)} # of det pulses:{len(dets)} true mu:{true_mu}")

plt.show()
#marginalise over std
d_pos = np.trapz(mat,std_arr,axis=1)
print(d_pos.shape)
plt.figure(figsize=(12,6),dpi=100)
fig = plt.pcolormesh(mu_arr,N_arr,d_pos.T)
ax = fig.axes
ax.set_ylim(min(N_arr),max(N_arr))
ax2 = ax.twinx()
f_arr = N_arr/(O/p)
ax2.set_ylim(min(f_arr),max(f_arr))
ax.set_xlabel(r'$\mu$')
ax.set_ylabel(r'$N_e$')
ax2.set_ylabel('f')
plt.title(f"simulated pulses(true frac):{len(snrs)}({f}) detected pulses:{len(dets)} true mu:{true_mu}")
plt.show()

d_pos = np.trapz(mat,mu_arr,axis=0)
print(d_pos.shape)
plt.figure(figsize=(12,6),dpi=100)
fig = plt.pcolormesh(std_arr,N_arr,d_pos.T)
ax = fig.axes
ax.set_ylim(min(N_arr),max(N_arr))
ax2 = ax.twinx()
ax2.set_ylim(min(f_arr),max(f_arr))

ax.set_xlabel(r'$\sigma$')
ax.set_ylabel(r'$N_e$')
ax2.set_ylabel('f')
plt.title(f"simulated pulses(true frac):{len(snrs)}({f}) detected pulses:{len(dets)} true std:{true_std}")
plt.show()

print(data.__dict__)
