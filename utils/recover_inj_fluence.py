#!/usr/bin/env python3

import numpy as np
from presto.filterbank import FilterbankFile
from presto import filterbank as fb
from presto import rfifind
import argparse
from matplotlib import pyplot as plt
from sigpyproc import readers as r
import copy
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

def maskfile(maskfn, data, start_bin, nbinsextra,mask_zero=True):
    from presto import rfifind
    print('loading mask')
    rfimask = rfifind.rfifind(maskfn)
    print('getting mask')
    mask = get_mask(rfimask, start_bin, nbinsextra)[::-1]
    print('get mask finished')
    masked_chans = mask.all(axis=1)
    #mask the data but set to the mean of the channel
    return data, masked_chans

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

def grab_spectra(gfb,ts_arr,te_arr,mask_fn_arr,dm):
     #load the filterbank file
    for gf,ts,te,mask_fn in zip(gfb,ts_arr,te_arr,mask_fn_arr):
        g = r.FilReader(gf)
        tsamp = float(g.header.tsamp)
        nsamps = int((te-ts)/tsamp)
        ssamps = int(ts/tsamp)
        # data = np.load("data.npz",allow_pickle=1)['data']

        spec = g.read_block(ssamps,nsamps)
        # spec[:,:] = data[:,ssamps:ssamps+nsamps]

        #load mask
        data, masked_chans = maskfile(mask_fn,spec,ssamps,nsamps,mask_zero=False)
        # data = spec
        # masked_chans = np.load("data.npz",allow_pickle=1)['masked_chans']
        #data.subband(256,subdm=dm,padval='median')
        downsamp = 1
        stats_window = 1.0
        p = (4,0.05,dm)
        d_fluence,p_toa,p_fluence,p_dm = calculate_FLUENCE_wrapper([p,stats_window,tsamp,downsamp,data,masked_chans])


def time_to_bin(t, sample_rate):
    """Return time as a bin number provided a sampling time"""
    return np.round(t / sample_rate).astype(int)

def calculate_FLUENCE_wrapper(X):
    p = X[0]
    stats_window = X[1]
    tsamp = X[2]
    downsamp = X[3]
    data = X[4]
    masked_chans = X[5]
    p_toa, p_fluence, p_dm = p
    plot_bin = time_to_bin(stats_window,tsamp)
    pulse_bin = time_to_bin(p_toa,tsamp)
    print(pulse_bin,plot_bin)
    #plotting
    new_d = copy.deepcopy(data)
    new_d = new_d.dedisperse(dm=p_dm)
    d = new_d[:,pulse_bin-plot_bin:pulse_bin+plot_bin]
    #reshape for downsampling
    end = d.shape[1]-d.shape[1]%int(downsamp)
    d = d[:,0:end]
    #dedisperse and downsample
    print(downsamp)
    d = d.downsample(tfactor = downsamp,ffactor=1)
    print(sum(masked_chans))
    d = d[~masked_chans,:]
    # plt.figure()
    # plt.imshow(d.normalise(),aspect="auto")
    # plt.show()
    ts = d.mean(axis=0)
    print("calculating fluence")
    d_fluence= calculate_FLUENCE([ts,tsamp*downsamp,6e-2,int(plot_bin/downsamp)])
    print(f"Inj fluence:{p_fluence} Det fluence: {d_fluence}")
    return d_fluence,p_toa,p_fluence,p_dm

def calculate_FLUENCE(X):
    #calculates the FLUENCE given a timeseries
    ts = X[0]
    tsamp = X[1]
    width = X[2]
    nsamp = X[3]
    w_bin = int(width/tsamp)
    x = np.linspace(0,tsamp*len(ts),len(ts))
    ts_std = np.delete(ts,range(int(nsamp-w_bin),int(nsamp+w_bin)))
    x_std = np.delete(x,range(int(nsamp-w_bin),int(nsamp+w_bin)))
    plt.plot(x,ts)
    plt.show()
    #fit a polynomial
    coeffs = np.polyfit(x_std,ts_std,10)
    poly = np.poly1d(coeffs)
    #subtract the mean
    ts_sub = ts-poly(x)
    ts_start = ts[:int(nsamp-w_bin)]
    x_start = x[:int(nsamp-w_bin)]
    ts_start = ts_start - poly(x_start)
    ts_end = ts[int(nsamp+w_bin):]
    x_end = x[int(nsamp+w_bin):]
    ts_end = ts_end - poly(x_end)

    fluence_noise = np.trapz(ts_start,x_start) + np.trapz(ts_end,x_end)
    plt.figure()
    plt.plot(x_start,ts_start)
    plt.plot(x_end,ts_end)
    print(fluence_noise)

    plt.show()
    #grab just the window
    ts_fluence = ts_sub[nsamp-w_bin:nsamp+w_bin]
    x_fluence = x[nsamp-w_bin:nsamp+w_bin]
    fluence = np.trapz(ts_fluence,x_fluence)
    fluence = np.trapz(ts_sub,x)
    plt.plot(x,ts_sub)
    plt.show()
    return fluence

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
    ts_arr = [t-4]
    te_arr = [t+4]
    print(mask_arr,ts_arr,te_arr)
    grab_spectra([bfb],ts_arr,te_arr,mask_arr,dm)
