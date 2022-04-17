#!/usr/bin/env python3

import numpy as np
from presto.filterbank import FilterbankFile
from presto import filterbank as fb
from presto import rfifind
import argparse
from matplotlib import pyplot as plt
parser = argparse.ArgumentParser(description='Process filterbank positions')
parser.add_argument('--base_fil',type=str,help="the filterbank file for the detection")
parser.add_argument('--t',type=float,help="start time for the observation in the base filterbank file")
parser.add_argument('--dm',type=float,help="dm in pc/cm^3")
args = parser.parse_args()

bfb = args.base_fil
t = args.t
dm = args.dm
#we assume that the filterbank files are processed with the automated filterbank pipeline, therefore we need to find the rfi mask file
def get_mask_fn(filterbank):
    folder = filterbank.strip('.fil')
    mask = f"{folder}_rfifind.mask"
    return mask

def get_mask(rfimask, startsamp, N):
    """Return an array of boolean values to act as a mask
        for a Spectra object.

        Inputs:
            rfimask: An rfifind.rfifind object
            startsamp: Starting sample
            N: number of samples to read

        Output:
            mask: 2D numpy array of boolean values.
                True represents an element that should be masked.
    """
    sampnums = np.arange(startsamp, startsamp+N)
    blocknums = np.floor(sampnums/rfimask.ptsperint).astype('int')
    mask = np.zeros((N, rfimask.nchan), dtype='bool')
    for blocknum in np.unique(blocknums):
        blockmask = np.zeros_like(mask[blocknums==blocknum])
        chans_to_mask = rfimask.mask_zap_chans_per_int[blocknum]
        if chans_to_mask.any():
            blockmask[:,chans_to_mask] = True
        mask[blocknums==blocknum] = blockmask
    return mask.T


def get_mask_arr(gfb):
    mask_arr = []
    for g in gfb:
        print(g)
        mask_arr.append(get_mask_fn(g))
    return mask_arr

#grab the time stamps for all the filterbank files
def get_timestamps(bfb,ts,te,gfb):
    #load the filterbank file
    f = FilterbankFile(bfb,mode='read')
    t_start = f.header['tstart']+(ts/(60*60*24))
    t_end = f.header['tstart']+(te/(60*60*24))

    ts_arr = []
    te_arr = []
    for gf in gfb:
        g = FilterbankFile(gf,mode='read')
        #append and convert to seconds
        ts_arr.append((t_start - g.header['tstart'])*60*60*24)
        te_arr.append((t_end - g.header['tstart'])*60*60*24)
    return ts_arr,te_arr

def maskfile(maskfn, data, start_bin, nbinsextra):
    from presto import rfifind
    rfimask = rfifind.rfifind(maskfn)
    mask = get_mask(rfimask, start_bin, nbinsextra)[::-1]
    masked_chans = mask.all(axis=1)
    data = data.masked(mask, maskval='median-mid80')
    return data, masked_chans

def grab_spectra(gfb,ts_arr,te_arr,mask_fn_arr,dm):
     #load the filterbank file
    for gf,ts,te,mask_fn in zip(gfb,ts_arr,te_arr,mask_fn_arr):
        g = FilterbankFile(gf,mode='read')
        tsamp = float(g.header['tsamp'])
        nsamps = int((te-ts)/tsamp)
        ssamps = int(ts/tsamp)
        #sampels to burst
        nsamps_start_zoom = int(2.5/tsamp)
        nsamps_end_zoom = int(3.5/tsamp)
        spec = g.get_spectra(ssamps,nsamps)
        #load mask
        spec.dedisperse(dm, padval='median')
        data, masked_chans = maskfile(mask_fn,spec,ssamps,nsamps)
        #data.subband(256,subdm=dm,padval='median')
        subband = 256
        downsamp = 1
        # data.downsample(int(downsamp))
        # data.subband(int(subband))
        # data = data.scaled(False)
        dat_arr = data.data
        dat_arr = dat_arr[~masked_chans,:]
        dat_arr = dat_arr[:,int(nsamps_start_zoom/downsamp):int(nsamps_end_zoom/downsamp)]
        dat_ts = np.mean(dat_arr,axis=0)

        SNR,ts_sub,std = calculate_SNR(dat_ts,tsamp,10e-3)
        print(SNR,std)
        plt.plot(ts_sub)
        plt.title(f"{gf} - SNR:{SNR}")
        # plt.figure()
        # plt.imshow(dat_arr,aspect='auto')
        plt.show()

def calculate_SNR(ts,tsamp,width):
    #calculates the SNR given a timeseries

    ind_max = np.argwhere(np.max(ts)==ts)
    w_bin = width/tsamp
    ts_std = np.delete(ts,range(int(ind_max-w_bin),int(ind_max+w_bin)))
    # ts_std = ts
    mean = np.median(ts_std)
    std = np.std(ts_std)
    #subtract the mean
    ts_sub = ts-mean
    #remove rms
    Amplitude = np.max(ts_sub) - np.sqrt(np.mean(ts_sub**2))
    snr = Amplitude/std
    # print(np.mean(ts_sub))
    # plt.plot(ts_std)
    # plt.show()
    #area of pulse, the noise, if gaussian should sum, to 0
    return snr,ts_sub,std

mask_arr = get_mask_arr([bfb])
ts_arr = [t-3]
te_arr = [t+3]
print(mask_arr,ts_arr,te_arr)
grab_spectra([bfb],ts_arr,te_arr,mask_arr,dm)
