#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import sys
data = np.load(sys.argv[1])

mat = data['data']
N_arr = data['N']
mu_arr = data['mu']
std_arr = data['std']
dets = data['det']
mat = mat-np.max(mat)
mat = np.exp(mat)
snrs = [1]
posterior = np.trapz(np.trapz(mat,mu_arr,axis=0),std_arr,axis=0)
print(posterior)
plt.figure()
plt.plot(N_arr,posterior)
plt.xlabel('N')
plt.title(f"# of simulated pulses:{len(snrs)} # of det pulses:{len(dets)}")
posterior = np.trapz(np.trapz(mat,std_arr,axis=1),N_arr,axis=1)
plt.figure()
plt.plot(mu_arr,posterior)
plt.xlabel('mu')
plt.title(f"# of simulated pulses:{len(snrs)} # of det pulses:{len(dets)}")

#marginalise over std
d_pos = np.trapz(mat,std_arr,axis=1)
print(d_pos.shape)


plt.figure()
plt.pcolormesh(mu_arr,N_arr,d_pos.T)
plt.xlabel('mu')
plt.ylabel('N')
plt.title(f"# of simulated pulses:{len(snrs)} # of det pulses:{len(dets)}")
plt.figure()
d_pos = np.trapz(mat,mu_arr,axis=0)
print(d_pos.shape)
plt.pcolormesh(std_arr,N_arr,d_pos.T)
plt.xlabel('sig')
plt.ylabel('N')
plt.title(f"# of simulated pulses:{len(snrs)} # of det pulses:{len(dets)}")
plt.show()

print(data.__dict__)
