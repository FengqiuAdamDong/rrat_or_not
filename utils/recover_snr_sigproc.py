#!/usr/bin/env python3

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
        g = r.FilReader(gf)
        tsamp = float(g.header.tsamp)
        nsamps = int((te-ts)/tsamp)
        ssamps = int(ts/tsamp)
        spec = g.read_block(ssamps,nsamps)
        #load mask
        # data, masked_chans = maskfile(mask_fn,spec,ssamps,nsamps)
        #data.subband(256,subdm=dm,padval='median')
        downsamp = 3
        stats_window = 0.75
        p = (3,5,dm)
        d_snr,a,s,p_toa,p_snr,p_dm = calculate_SNR_wrapper([p,stats_window,tsamp,downsamp,spec])


def time_to_bin(t, sample_rate):
    """Return time as a bin number provided a sampling time"""
    return np.round(t / sample_rate).astype(int)

def calculate_SNR_wrapper(X):
    p = X[0]
    stats_window = X[1]
    tsamp = X[2]
    downsamp = X[3]
    data = X[4]
    p_toa, p_snr, p_dm = p
    plot_bin = time_to_bin(stats_window,tsamp)
    plot_bin = plot_bin - (plot_bin%downsamp)
    pulse_bin = time_to_bin(p_toa,tsamp)
#     #plotting
    new_d = copy.deepcopy(data)
    stats = copy.deepcopy(new_d[:,pulse_bin-plot_bin:pulse_bin+plot_bin])
    stats = stats.downsample(downsamp)
    std_ndds = np.std(np.mean(stats,axis=0))
    new_d = new_d.dedisperse(dm=p_dm)
    d = new_d[:,pulse_bin-plot_bin:pulse_bin+plot_bin]
    d = d.downsample(downsamp)
    ts = d.mean(axis=0)
    print("calculating snr")
    #make an array for this
    d_snr,a,s = calculate_SNR([ts,tsamp*downsamp,std_ndds,1e-2,int(plot_bin/downsamp)])
    print(f"Inj snr:{p_snr} Det snr: {d_snr} Amplitude:{a} std:{s} std_ndds:{std_ndds}")
    return d_snr,a,s,p_toa,p_snr,p_dm

def calculate_SNR(X):
    #calculates the SNR given a timeseries
    ts = X[0]
    tsamp = X[1]
    std_ndds = X[2]
    width = X[3]
    nsamp = X[4]
    ind_max = nsamp
    w_bin = width/tsamp
    try:
        # ts_std = np.delete(ts,range(int(ind_max-w_bin),int(ind_max+w_bin)))
        ts_std = ts
    except:
        print("ENCOUNTERED ERROR ***")
        import pdb; pdb.set_trace()
    # ts_std = ts
    mean = np.median(ts_std)
    std = np.std(ts_std)
    #subtract the mean
    ts_sub = ts-mean
    #remove rms
    Amplitude = ts_sub[nsamp]
    snr = Amplitude/std
    # print(np.mean(ts_sub))
    plt.plot(ts_std)
    plt.show()
    #area of pulse, the noise, if gaussian should sum, to 0
    return snr,Amplitude,std

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
