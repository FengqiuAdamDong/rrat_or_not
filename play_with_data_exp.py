#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import sys
data = np.load(sys.argv[1])

mat = data['data']
N_arr = data['N']
snrs = data['snrs']
dets = data['det']
true_k = data['true_k']
k_arr = data['k']
mat = mat-np.max(mat)
mat = np.exp(mat)
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

print(data.__dict__)
