#!/usr/bin/env python

import os
import itertools
import argparse
import numpy as np
import matplotlib

# matplotlib.use("pdf")
import matplotlib.pyplot as plt
from presto.filterbank import FilterbankFile, create_filterbank_file
from presto import spectra as spec
import copy
from multiprocessing import Pool
import sigpyproc
from sigpyproc import utils as u
import sys
from inject_stats import autofit_pulse
# define the random number generator
# np.random.seed(12345)

# TRIAL_SNR = [10]
# 0.8,
# ]  # total S/N of pulse
# TRIAL_SNR = np.linspace(0.1,4,30)
TRIAL_SNR=[1,2,3,4,5,6]
# TRIAL_SNR = [0.003]
# TRIAL_SNR = np.linspace(0.1e-3, 5e-3, 50)
pulse_width = [12.41e-3]  # 10ms pulse width

TRIAL_DMS = [
    100,
]  # DM of bursts

DM_CONST = 4149.377593360996  # dispersion constant


def dm_delay(dm, f1, f2):
    """Return DM delay in seconds"""
    return DM_CONST * dm * (1.0 / f2**2 - 1.0 / f1**2)


def time_to_bin(t, sample_rate):
    """Return time as a bin number provided a sampling time"""
    return np.round(t / sample_rate).astype(int)

def create_pulse_attributes(npulses=1, duration=200, min_sep=2):
    """
    For a given number of pulses over a certain time duration,
    create an injection sample based on the pre-determined list
    of S/N and DM values.
    """
    max_dm_delay = dm_delay(max(TRIAL_DMS), 800, 400)

    toas = np.zeros(npulses) + duration
    while max(toas) > (duration - max_dm_delay):
        dtoa = np.zeros(npulses + 1)
        while min(dtoa) < min_sep:
            # don't inject anything into the last 10% of data
            centre_pulse = duration / 2
            toas = (
                (np.linspace(0, npulses, npulses) - int(npulses / 2)) * min_sep
            ) + centre_pulse
            dtoa = np.diff(toas)
            if (min(toas) < (5)) | (max(toas) > (duration - 5)):
                print("too many pulses for length, toas:", toas)
                sys.exit(1)
            if len(toas) == 1:
                break
    # get the cartesian product so that we have a generator of TOA x SNR x DM
    grid_coords = np.array(
        [*itertools.product(toas, TRIAL_SNR, TRIAL_DMS, pulse_width)]
    )

    # save the injection sample to a numpy file
    print("saving injection sample to: sample_injections.npy")
    downsamp = 3
    stats_window = 1
    np.savez(
        "sample_injections",
        grid=grid_coords,
        downsamp=downsamp,
        stats_window=stats_window,
    )

    return (
        grid_coords,
        npulses,
        len(TRIAL_SNR),
        len(TRIAL_DMS),
        downsamp,
        stats_window,
    )


def adjusted_peak(desired_a, tsamp, sigma, ds):
    # calculate the adjusted peak height after downsampling
    # width in bins
    width_bins = sigma / tsamp
    sum_term = 0
    for i in range(int(ds)):
        sum_term += np.exp(-((-i) ** 2) / (2 * width_bins))
    new_peak = sum_term / ds
    new_amplitude = desired_a / new_peak
    return new_amplitude


def inject_pulses(
    data, masked_chans, header, freqs, pulse_attrs, downsamp, stats_window, plot=False
):
    """For a given set of pulses, inject them into sample data"""
    # get the noise level in each channel
    # print("estimating initial noise values pre-injection")
    # per_chan_noise = np.std(data, axis=1)
    tsamp = header.tsamp
    statistics = []
    # data[:,:] = 0
    # data_copy = copy.deepcopy(data)
    # 10s stats window
    for i, p in enumerate(pulse_attrs):
        p_toa, p_SNR, p_dm, p_width = p
        print("computing toas per channel")
        # start the pulse 100 samples after the first simulated time step
        toa_bin_top = 500
        # assume TOA is arrival time at top of band
        max_dm_delay = dm_delay(p_dm, max(freqs), min(freqs))
        print(f"max dm delay {max_dm_delay}")
        max_dm_delay_bins = time_to_bin(max_dm_delay, tsamp)
        print(f"dm delay across band = {max_dm_delay} s = {max_dm_delay_bins} bins")
        nbins_to_sim = max_dm_delay_bins + 2 * toa_bin_top
        x = np.linspace(0, nbins_to_sim, nbins_to_sim)

        # pulse peak time at each frequency
        dm_delays = dm_delay(p_dm, freqs[0], freqs)
        per_chan_toa_bins = toa_bin_top + time_to_bin(dm_delays, tsamp)

        # get the data to do statistics
        if p_toa < stats_window:
            stats_window = p_toa

        SNR, amp, stats_std,loc,sigma_width =  calculate_SNR_wrapper(p,stats_window,tsamp,downsamp,data,masked_chans,plot)
        print(f"stats std: {stats_std}")
        # calculate off pulse mean
        print("calculating expected S/N per channel")
        # convert S/N into actual power value
        # simulate the pulse as a Gaussian, normalise such that the
        # peak corresponds to the per-channel power level corresponding
        # to the total S/N

        width_bins = time_to_bin(p_width, tsamp)
        total_inj_pow = p_SNR * stats_std
        print(f"total power:{total_inj_pow}")

        pulse_wf = np.exp(
            -((x - per_chan_toa_bins[:, np.newaxis]) ** 2) / (2 * width_bins**2)
        )
        print("rescaling pulses to correct amplitude")
        pulse_wf /= pulse_wf.max(axis=1)[:, np.newaxis]
        pulse_wf *= total_inj_pow
        # x = np.linspace(0, pulse_wf.shape[1], pulse_wf.shape[1]) * tsamp
        # plt.plot(np.trapz(pulse_wf,x))
        # plt.show()

        # ensure that everything is shifted correctly when changes to uint8
        # just run on 32 bit data, whatever
        pulse_wf += np.random.rand(pulse_wf.shape[0], pulse_wf.shape[1]) - 0.5
        pulse_wf[pulse_wf < 0] = 0
        pulse_wf = np.around(pulse_wf)
        # combining the pulse with data
        print("combining simulated pulse with data")
        true_start_bin = time_to_bin(p_toa, tsamp) - toa_bin_top
        true_end_bin = true_start_bin + pulse_wf.shape[1]
        print(f"start bin = {true_start_bin}  end bin = {true_end_bin}")
        data[:, true_start_bin:true_end_bin] += pulse_wf
    # data = data.astype("uint8")
    # np.savez('data',data = data,masked_chans = masked_chans)
    # for i, p in enumerate(pulse_attrs):
    # downsamp=1
        # statistics.append(calculate_SNR_wrapper(p,stats_window,tsamp,downsamp,copy.deepcopy(data),masked_chans))
    # for i, p in enumerate(pulse_attrs):
    # downsamp=1
    # statistics.append(calculate_SNR_wrapper([p,stats_window,tsamp,downsamp,data_copy,masked_chans]))

    return data, statistics


def calculate_SNR_wrapper(p,stats_window,tsamp,downsamp,data,masked_chans,plot):
    p_toa, p_SNR, p_dm, p_width = p
    stats_start, stats_end = (
        time_to_bin(p_toa - stats_window, tsamp * downsamp),
        time_to_bin(p_toa + stats_window, tsamp * downsamp),
    )
    stats_data = copy.deepcopy(data)
    #dedisperse stats data
    stats_data = stats_data.dedisperse(p_dm)
    #downsample stats_data to the downsample that I will use for detection
    print(f"downmsampling with {downsamp}")
    stats_data = stats_data.downsample(tfactor=downsamp)
    stats_data = stats_data[~masked_chans,:]
    #get the window
    stats_data = stats_data[:, stats_start:stats_end]

    #fit a polynomial to the window
    #get the mean of the window
    stats_mean = np.mean(stats_data, axis=0)
    SNR,amp,std,loc,sigma_width =  autofit_pulse(stats_mean,tsamp*downsamp,p_width*6,int(stats_window/tsamp/downsamp)
                                                    ,data,downsamp,plot=plot)
    print(f"Inj SNR:{p_SNR} Det SNR: {SNR} std: {std} amp: {amp} loc: {loc} width: {sigma_width}")
    return SNR, amp, std,loc,sigma_width


def maskfile(maskfn, data, start_bin, nbinsextra):
    from presto import rfifind

    print("loading mask")
    rfimask = rfifind.rfifind(maskfn)
    print("getting mask")
    mask = get_mask(rfimask, start_bin, nbinsextra)[::-1]
    print("get mask finished")
    masked_chans = mask.all(axis=1)
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
    sampnums = np.arange(startsamp, startsamp + N)
    blocknums = np.floor(sampnums / rfimask.ptsperint).astype("int")
    mask = np.zeros((N, rfimask.nchan), dtype="bool")
    for blocknum in np.unique(blocknums):
        blockmask = np.zeros_like(mask[blocknums == blocknum])
        chans_to_mask = rfimask.mask_zap_chans_per_int[blocknum]
        if chans_to_mask.any():
            blockmask[:, chans_to_mask] = True
        mask[blocknums == blocknum] = blockmask
    return mask.T


def get_pazi_mask(pazi_fn):
    import psrchive
    arch = psrchive.Archive_load(pazi_fn)
    #lets get the mask
    total_subint = arch.get_nsubint()
    total_chans = arch.get_nchan()
    total_bins = arch.get_nbin()
    mask_array = np.zeros((total_subint,1,total_chans,total_bins))
    for i,integration in enumerate(arch):
        for j in range(integration.get_nchan()):
            mask_array[i,0,j,:] = integration.get_weight(j)
    masks = np.mean(np.mean(np.mean(mask_array,axis=2),axis=1),axis=1)
    return masks


def get_filterbank_data_window(fn, maskfn, duration=20):
    """
    Open the filterbank file, extract the spectra around
    the middle (+/- duration / 2) and return that data as a
    2D array.
    """
    from sigpyproc import readers as r
    from sigpyproc.block import FilterbankBlock as fbb
    #find the archive filename
    pazi_fn = fn.replace(".fil", "") + "_1000.00ms_Cand.pfd.pazi"
    #load the archive
    pazi_mask = get_pazi_mask(pazi_fn)
    #load the weights
    print("getting filterbank data")
    filf = r.FilReader(fn)
    hdr = filf.header
    tsamp = hdr.tsamp
    chunk_size = hdr.nsamples//len(pazi_mask)
    chunk_dur = chunk_size * tsamp
    #get required chunks
    reuired_chunks = duration/chunk_dur
    fil_dur = hdr.nsamples * tsamp
    # start in the middle of the data
    start_chunk = int(len(pazi_mask)//2 - reuired_chunks//2)
    #recursively get the data and throw away if chunk is masked
    acquired_chunks = 0
    current_chunk = int(start_chunk)
    channel_mask = np.zeros(hdr.nchans,dtype=bool)
    while acquired_chunks < reuired_chunks:
        #check if chunk is masked
        if pazi_mask[current_chunk] == 0:
            current_chunk += 1
            continue
        #get the data for this chunk
        start_bin = current_chunk * chunk_size
        _ = filf.read_block(start_bin, chunk_size)
        masked_data, masked_chans = maskfile(maskfn, copy.deepcopy(_), start_bin, chunk_size)
        #update the channel mask
        channel_mask = np.logical_or(channel_mask,masked_chans)
        #update the chunk counter
        if acquired_chunks>0:
            data = np.append(data,_,axis=1)
        else:
            data = _
        acquired_chunks += 1
        current_chunk += 1
    # update the header so that it represents the windowed data
    presto_header = FilterbankFile(fn).header
    hdr.tstart += (start_chunk * chunk_size * tsamp) / 86400.0  # add in MJD
    hdr.nsamples = data.shape[1]  # total number of data samples (nchan * nspec)
    hdr.nsamples_files = [data.shape[1]]
    hdr.tstart_files = [hdr.tstart]
    data = fbb(data, hdr)
    #don't bother updating the presto header
    return data, channel_mask, presto_header


def multiprocess(arr):
    (
        dm,
        s,
        ifn,
        duration,
        maskfn,
        injection_sample,
        stats_window,
        downsamp,
        single_SNR_dm,
    ) = arr
    rawdata, masked_chans, presto_header = get_filterbank_data_window(
        ifn, duration=duration, maskfn=maskfn
    )
    p_arr = (
        dm,
        s,
        ifn,
        duration,
        maskfn,
        injection_sample,
        rawdata,
        masked_chans,
        presto_header,
        downsamp,
        stats_window,
        single_SNR_dm,
    )
    process(p_arr)


def process(pool_arr):
    # this SNR field is only for the filename purposes, so you can pass a string into here
    (
        dm,
        SNR,
        ifn,
        duration,
        maskfn,
        injection_sample,
        rawdata_,
        masked_chans_,
        presto_header,
        downsamp,
        stats_window,
        single_SNR_dm,
    ) = pool_arr
    rawdata = copy.deepcopy(rawdata_)
    masked_chans = copy.deepcopy(masked_chans_)
    print(f"dm={dm} SNR={SNR} ds={downsamp} stats_window={stats_window}")
    # figure out what pulses to add
    if single_SNR_dm:
        add_mask = np.logical_and(
            injection_sample[:, 2] == dm, injection_sample[:, 1] == SNR
        )

        pulses_to_add = injection_sample[add_mask]
    else:
        pulses_to_add = injection_sample

    header = rawdata.header
    freqs = np.array(range(header.nchans)) * header.foff + header.fch1
    injdata, statistics = inject_pulses(
        rawdata, masked_chans, header, freqs, pulses_to_add, downsamp, stats_window
    )
    s = np.array(statistics)
    SNR_4 = str(np.around(SNR, 4)).zfill(6)
    ofn = os.path.basename(ifn).replace(".fil", f"_inj_dm{dm}_SNR{SNR_4}.fil")
    print(f"creating output file: {ofn}")
    presto_header["nbits"] = 8
    create_filterbank_file(
        ofn, presto_header, nbits=presto_header["nbits"], spectra=injdata.T
    )

    # NOTE: the filterbank spectra need to be provided with shape (nspec x nchan),
    # so we have to transpose the injected array at write time.
    # injdata.to_file(ofn)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("fil", help="Filterbank file to use as raw data background")
    parser.add_argument("--m", help="this is the mask fn")
    parser.add_argument(
        "--d", help="Duration of required output file", type=float, default=300
    )
    parser.add_argument("--n", help="Number of pulses to inject", type=int, default=50)
    parser.add_argument(
        "-F",
        "--fake",
        help="create fake data set with gaussian noise properties and no RFI",
        action="store_true",
    )
    parser.add_argument(
        "-i",
        "--injection_file",
        help="pre-made injections .npy file to use when injecting pulses",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    duration = args.d
    ifn = args.fil
    maskfn = args.m
    if args.injection_file:
        injection_sample = np.load(args.injection_file)
        npul = len(injection_sample)
        ndm = len(set(injection_sample[:, 2]))
        nSNR = len(set(injection_sample[:, 1]))
    else:
        (
            injection_sample,
            npul,
            nSNR,
            ndm,
            downsamp,
            stats_window,
        ) = create_pulse_attributes(npulses=args.n, duration=duration)
    print(f"total number of injections: {len(injection_sample)}")

    print(f"number of injection DMs: {ndm}")
    print(f"number of injection SNRs: {nSNR}")

    # header, freqs, rawdata,masked_data = get_filterbank_data_window(ifn, duration=duration,maskfn=maskfn)
    print(f"getting data cutout from: {ifn}")
    # add the pulses
    multiprocessing = True
    if multiprocessing:
        for dm in TRIAL_DMS:
            pool_arr = []
            for s in TRIAL_SNR:
                pool_arr.append(
                    (
                        dm,
                        s,
                        ifn,
                        duration,
                        maskfn,
                        injection_sample,
                        stats_window,
                        downsamp,
                        True,
                    )
                )
            with Pool(10) as p:
                p.map(multiprocess, pool_arr)
    else:
        rawdata, masked_chans, presto_header = get_filterbank_data_window(
            ifn, duration=duration, maskfn=maskfn
        )
        for dm in TRIAL_DMS:
            pool_arr = []
            for s in TRIAL_SNR:
                pool_arr.append(
                    (
                        dm,
                        s,
                        ifn,
                        duration,
                        maskfn,
                        injection_sample,
                        rawdata,
                        masked_chans,
                        presto_header,
                        downsamp,
                        stats_window,
                        True,
                    )
                )
            for p in pool_arr:
                process(p)
