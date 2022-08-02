#!/usr/bin/env python3

import numpy as np
from presto.filterbank import FilterbankFile
from presto import filterbank as fb
from presto import rfifind
import argparse
from matplotlib import pyplot as plt

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
        downsamp = 3
        #subtract running median and divide by std
        chunk = 0.5
        #do this so that chunk_samp will always be a mutliple of 2
        chunk_samp = int(chunk/2/tsamp/downsamp)*2



        data.downsample(int(downsamp))
        dat_arr = data.data
        samp_low = int(2.75/tsamp/downsamp)
        samp_high = int(3.25/tsamp/downsamp)
        import copy
        _temp = copy.deepcopy(dat_arr)
        for i in range(dat_arr.shape[1]):
            half_chunk = int(chunk_samp/2)
            if ( i>samp_low ) & ( i<samp_high ):
                pass
            else:
                if i<=half_chunk:
                    #if it's less than half then just forward cast
                    stat_d = dat_arr[:,0:chunk_samp]
                elif (i>(samp_low-half_chunk)) & (i<samp_low):
                    stat_d = dat_arr[:,(samp_low-chunk_samp):samp_low]
                elif ( i>samp_high ) & (i<( samp_high+half_chunk )):
                    stat_d = dat_arr[:,samp_high:samp_high+chunk_samp ]
                elif i>=nsamps-1-half_chunk:
                    stat_d = dat_arr[:,nsamps-1-chunk_samp:nsamps-1]
                else:
                    stat_d = dat_arr[:,i-half_chunk:i+half_chunk]

                med = np.median(stat_d,axis=1)
                std = np.std(stat_d,axis=1)

            std[std==0]=1
            _temp[:,i] = (_temp[:,i]-med)/std
        dat_arr = _temp
        data.data = dat_arr
        dat_arr = data.data
        dat_arr = dat_arr[~masked_chans,:]
        dat_arr = dat_arr[:,int(nsamps_start_zoom/downsamp):int(nsamps_end_zoom/downsamp)]
        dat_ts = np.mean(dat_arr,axis=0)

        SNR,ts_sub,std,max_samp = calculate_SNR(dat_ts,tsamp*downsamp,10e-2,nsamps=int(0.5/tsamp/downsamp))




        plt.figure()
        plt.plot(ts_sub)
        plt.scatter(max_samp,ts_sub[max_samp],marker="x",s=1000,c="red")
        plt.title(f"{gf} - SNR:{SNR}")
        plt.figure()
        plt.imshow(dat_arr,aspect='auto')
        plt.show()

def calculate_SNR(ts,tsamp,width,nsamps):
    #calculates the SNR given a timeseries

    ind_max = nsamps
    w_bin = int(width/tsamp)
    ts_std = np.delete(ts,range(int(ind_max-w_bin),int(ind_max+w_bin)))

    # ts_std = ts
    mean = np.median(ts_std)
    std = np.std(ts_std)
    #subtract the mean
    ts_sub = ts-mean
    ts_max = ts_sub[nsamps-w_bin:nsamps+w_bin]
    #remove rms
    Amplitude = max(ts_max)
    snr = Amplitude/std
    print(Amplitude,std,snr)
    # print(np.mean(ts_sub))
    # plt.plot(ts_std)
    # plt.show()
    #area of pulse, the noise, if gaussian should sum, to 0
    max_samp = np.argwhere(ts_sub == Amplitude)
    return snr,ts_sub,std,max_samp

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process filterbank positions')
    parser.add_argument('--base_fil',type=str,help="the filterbank file for the detection")
    parser.add_argument('--t',type=float,help="start time for the observation in the base filterbank file")
    parser.add_argument('--dm',type=float,help="dm in pc/cm^3")
    args = parser.parse_args()

    bfb = args.base_fil
    t = args.t
    dm = args.dm
    mask_arr = get_mask_arr([bfb])
    ts_arr = [t-3]
    te_arr = [t+3]
    print(mask_arr,ts_arr,te_arr)
    grab_spectra([bfb],ts_arr,te_arr,mask_arr,dm)
