#!/usr/bin/env python3
import numpy as np
from presto.filterbank import FilterbankFile
from presto import filterbank as fb
from presto import rfifind
import argparse
from matplotlib import pyplot as plt
from sigpyproc import readers as r
import copy
from scipy import signal

def grab_spectra(g,ts,te,dm=100,downsamp=4):
     #load the filterbank file
        g = r.FilReader(g)
        tsamp = float(g.header.tsamp)
        nsamps = int((te-ts)/tsamp)
        nsamps = nsamps - (nsamps%downsamp)
        ssamps = int(ts/tsamp)
        spec = g.read_block(ssamps,nsamps)
        #load mask
        # data, masked_chans = maskfile(mask_fn,spec,ssamps,nsamps)
        #data.subband(256,subdm=dm,padval='median')
        spec = spec.dedisperse(dm)
        spec = spec.downsample(downsamp)
        spec = spec.mean(axis=0)
        return spec

fil1 = 'B1905+39_sp_59161_pow.fil'
fil2 = 'B1905+39_sp_59161_pow_inj_dm100_snr10.fil'
ts_fil1 = 539.4206592
te_fil1 = 549.4206592
ts_fil2 = 0
te_fil2 = 10

spec1 = grab_spectra(fil1,ts_fil1,te_fil1,100)
spec2 = grab_spectra(fil2,ts_fil2,te_fil2,100)
plt.figure()
plt.plot(spec1)
plt.plot(spec2)
#cut out the pulse
spec2 = np.delete(spec2,range(np.argmax(spec2)-20,np.argmax(spec2)+20))
plt.figure()
plt.plot(spec1)
plt.plot(spec2)
plt.show()

# spec3 = np.load('data.npy')
# spec3 = np.mean(spec3,0)
# spec3 = spec3[0:len(spec3)-len(spec3)%4]
# correlate = signal.correlate(spec1,spec2)
# lags = signal.correlation_lags(len(spec1),len(spec2))
# truth = spec1==spec2
# plt.figure()
# plt.plot(truth)
# plt.show()
print(np.std(spec1))
print(np.std(spec2))
import pdb; pdb.set_trace()
# print(correlate)
# plt.figure()
# plt.plot(spec1)
# plt.figure()
# plt.plot(spec2)
# plt.show()
