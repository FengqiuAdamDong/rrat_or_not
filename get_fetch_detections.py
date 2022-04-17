#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
fn = sys.argv[1]
flist = os.listdir(fn)
snr_arr = []
mjd_arr = []
for f in flist:
    if 'h5' in f:
        #if the file is a picture get the file name
        snr = f.split('_')[-1]
        mjd = f.split('_')[1]
        print(mjd=='cand')
        if mjd=='cand':
            mjd = f.split('_')[2]
        snr_arr.append(float(snr.split('.h5')[0]))
        mjd_arr.append(int(mjd))
np.savez('snr_mjd_arr',snr=np.array(snr_arr),mjd=np.array(mjd_arr))
print(np.mean(snr_arr),np.median(snr_arr),np.std(np.log10(snr_arr)))
plt.figure()
plt.hist(snr_arr,bins=100)
plt.figure()
plt.hist(mjd_arr,bins=100)
plt.show()
