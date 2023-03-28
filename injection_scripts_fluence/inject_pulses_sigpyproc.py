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

# define the random number generator
# np.random.seed(12345)

# TRIAL_FLUENCE = [10]
# 0.8,
# ]  # total S/N of pulse
# TRIAL_FLUENCE = np.linspace(0.1,4,30)
# TRIAL_FLUENCE=[2,3,4,5,6,7,8,9,10]
TRIAL_FLUENCE = [0.003]
# TRIAL_FLUENCE = np.linspace(7e-4, 2.5e-3, 30)
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


def scale_to_uint8(a):
    """
    Rescale an array to fit within the dynamic range
    available to uint8. Shifts most negative value to 0.
    """
    mn = a.min()
    mx = a.max()

    mx -= mn

    a = ((a - mn) / mx) * 255
    return a.astype(np.uint8)


def create_pulse_attributes(npulses=1, duration=200, min_sep=2):
    """
    For a given number of pulses over a certain time duration,
    create an injection sample based on the pre-determined list
    of S/N and DM values.
    """
    max_dm_delay = dm_delay(max(TRIAL_DMS), 800, 400)

    toas = np.zeros(npulses) + duration
    pulse_width = [10e-3] #10ms pulse width
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
    # get the cartesian product so that we have a generator of TOA x FLUENCE x DM
    grid_coords = np.array([*itertools.product(toas, TRIAL_FLUENCE, TRIAL_DMS,pulse_width)])

    # save the injection sample to a numpy file
    print("saving injection sample to: sample_injections.npy")
    downsamp = 1
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
        len(TRIAL_FLUENCE),
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
    print(pulse_attrs)
    # 10s stats window
    for i, p in enumerate(pulse_attrs):
        p_toa, p_fluence, p_dm,p_width = p
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

        stats_start, stats_end = (
            time_to_bin(p_toa - stats_window, tsamp * downsamp),
            time_to_bin(p_toa + stats_window, tsamp * downsamp),
        )
        stats_nsamp = stats_end - stats_start
        print(f"injection TOA:{p_toa}")

        # calculate off pulse mean
        print("calculating expected S/N per channel")
        # convert S/N into actual power value
        # simulate the pulse as a Gaussian, normalise such that the
        # peak corresponds to the per-channel power level corresponding
        # to the total S/N

        width_bins = time_to_bin(p_width, tsamp)
        total_inj_pow = p_fluence / (width_bins * tsamp) / np.sqrt(2 * np.pi)
        print(f"total power:{total_inj_pow}")

        pulse_wf = np.exp(
            -((x - per_chan_toa_bins[:, np.newaxis]) ** 2) / (2 * width_bins**2)
        )
        print("rescaling pulses to correct amplitude")
        pulse_wf /= pulse_wf.max(axis=1)[:, np.newaxis]
        pulse_wf *= total_inj_pow
        x = np.linspace(0, pulse_wf.shape[1], pulse_wf.shape[1]) * tsamp
        # plt.plot(np.trapz(pulse_wf,x))
        # plt.show()

        # ensure that everything is shifted correctly when changes to uint8
        # just run on 32 bit data, whatever
        pulse_wf += (np.random.rand(pulse_wf.shape[0],pulse_wf.shape[1])-0.5)
        pulse_wf[pulse_wf<0]=0
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
    # statistics.append(calculate_FLUENCE_wrapper([p,stats_window,tsamp,downsamp,copy.deepcopy(data),masked_chans]))
    # for i, p in enumerate(pulse_attrs):
    # downsamp=1
    # statistics.append(calculate_FLUENCE_wrapper([p,stats_window,tsamp,downsamp,data_copy,masked_chans]))

    return data, statistics


def calculate_FLUENCE_wrapper(X):
    p = X[0]
    stats_window = X[1]
    tsamp = X[2]
    downsamp = X[3]
    data = X[4]
    masked_chans = X[5]
    p_toa, p_fluence, p_dm = p
    plot_bin = time_to_bin(stats_window, tsamp)
    pulse_bin = time_to_bin(p_toa, tsamp)
    # plotting
    print(sum(masked_chans))
    new_d = copy.deepcopy(data)
    new_d = new_d.dedisperse(dm=p_dm)
    d = new_d[:, pulse_bin - plot_bin : pulse_bin + plot_bin]
    print(pulse_bin, plot_bin)
    # reshape for downsampling
    end = d.shape[1] - d.shape[1] % int(downsamp)
    d = d[:, 0:end]
    # dedisperse and downsample
    d = d.downsample(tfactor=downsamp, ffactor=1)
    d = d[~masked_chans, :]
    # plt.figure()
    # plt.imshow(d.normalise(),aspect="auto")
    ts = d.mean(axis=0)
    print("calculating fluence")
    d_fluence = calculate_FLUENCE(
        [ts, tsamp * downsamp, 6e-2, int(plot_bin / downsamp)]
    )
    print(f"Inj fluence:{p_fluence} Det fluence: {d_fluence}")
    return d_fluence, p_toa, p_fluence, p_dm


def calculate_FLUENCE(X):
    # calculates the FLUENCE given a timeseries
    ts = X[0]
    tsamp = X[1]
    width = X[2]
    nsamp = X[3]
    w_bin = int(width / tsamp)
    x = np.linspace(0, tsamp * len(ts), len(ts))
    ts_std = np.delete(ts, range(int(nsamp - w_bin), int(nsamp + w_bin)))
    x_std = np.delete(x, range(int(nsamp - w_bin), int(nsamp + w_bin)))
    plt.plot(x, ts)
    plt.show()
    # fit a polynomial
    coeffs = np.polyfit(x_std, ts_std, 10)
    poly = np.poly1d(coeffs)
    # subtract the mean
    ts_sub = ts - poly(x)
    ts_start = ts[: int(nsamp - w_bin)]
    x_start = x[: int(nsamp - w_bin)]
    ts_start = ts_start - poly(x_start)
    ts_end = ts[int(nsamp + w_bin) :]
    x_end = x[int(nsamp + w_bin) :]
    ts_end = ts_end - poly(x_end)

    fluence_noise = np.trapz(ts_start, x_start) + np.trapz(ts_end, x_end)
    # grab just the window
    ts_fluence = ts_sub[nsamp - w_bin : nsamp + w_bin]
    x_fluence = x[nsamp - w_bin : nsamp + w_bin]
    fluence = np.trapz(ts_fluence, x_fluence)
    fluence = np.trapz(ts_sub, x)
    plt.plot(x, ts_sub)
    plt.show()
    return fluence


def maskfile(maskfn, data, start_bin, nbinsextra):
    from presto import rfifind

    print("loading mask")
    rfimask = rfifind.rfifind(maskfn)
    print("getting mask")
    mask = get_mask(rfimask, start_bin, nbinsextra)[::-1]
    print("get mask finished")
    masked_chans = mask.all(axis=1)
    # mask the data but set to the mean of the channel
    mask_vals = np.median(data, axis=1)
    # we used to need to mask files, not anymore though pass the data straight through
    # for i in range(len(mask_vals)):
    #     _ = data[i, :]
    #     _m = mask[i, :]
    #     _[_m] = mask_vals[i]
    #     data[i, :] = _
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


def get_filterbank_data_window(fn, maskfn, duration=20, masked_data=1):
    """
    Open the filterbank file, extract the spectra around
    the middle (+/- duration / 2) and return that data as a
    2D array.
    """
    from sigpyproc import readers as r

    print("getting filterbank data")
    filf = r.FilReader(fn)
    hdr = filf.header
    tsamp = hdr.tsamp
    fil_dur = hdr.nsamples * tsamp
    # start in the middle of the data
    start = fil_dur / 2 - duration / 2
    stop = start + duration
    print("start stop bins", start, stop)
    start_bin = int(np.round(start / tsamp))
    stop_bin = int(np.round(stop / tsamp))
    nsamp = stop_bin - start_bin
    # get the data
    _ = filf.read_block(start_bin, nsamp)
    # read the block
    if masked_data == None:
        print("removing masked_channels")
        masked_data, masked_chans = maskfile(maskfn, copy.deepcopy(_), start_bin, nsamp)
        masked_data = masked_data[~masked_chans, :]

    else:
        masked_data, masked_chans = maskfile(maskfn, copy.deepcopy(_), start_bin, nsamp)
    # update the header so that it represents the windowed data
    presto_header = FilterbankFile(fn).header
    hdr.tstart += (start_bin * tsamp) / 86400.0  # add in MJD
    hdr.nsamples = masked_data.shape[1]  # total number of data samples (nchan * nspec)
    return _, masked_data, masked_chans, presto_header


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
        single_fluence_dm,
    ) = arr
    rawdata, masked_data, masked_chans, presto_header = get_filterbank_data_window(
        ifn, duration=duration, maskfn=maskfn, masked_data=None
    )
    p_arr = (
        dm,
        s,
        ifn,
        duration,
        maskfn,
        injection_sample,
        rawdata,
        masked_data,
        masked_chans,
        presto_header,
        downsamp,
        stats_window,
        single_fluence_dm,
    )
    process(p_arr)


def process(pool_arr):
    # this FLUENCE field is only for the filename purposes, so you can pass a string into here
    (
        dm,
        fluence,
        ifn,
        duration,
        maskfn,
        injection_sample,
        rawdata_,
        masked_data_,
        masked_chans_,
        presto_header,
        downsamp,
        stats_window,
        single_fluence_dm,
    ) = pool_arr
    rawdata = copy.deepcopy(rawdata_)
    masked_chans = copy.deepcopy(masked_chans_)
    print(f"dm={dm} fluence={fluence} ds={downsamp} stats_window={stats_window}")
    # figure out what pulses to add
    if single_fluence_dm:
        add_mask = np.logical_and(
            injection_sample[:, 2] == dm, injection_sample[:, 1] == fluence
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
    fluence_4 = str(np.around(fluence, 4)).zfill(6)
    ofn = os.path.basename(ifn).replace(".fil", f"_inj_dm{dm}_fluence{fluence_4}.fil")
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
        nfluence = len(set(injection_sample[:, 1]))
    else:
        (
            injection_sample,
            npul,
            nfluence,
            ndm,
            downsamp,
            stats_window,
        ) = create_pulse_attributes(npulses=args.n, duration=duration)
    print(f"total number of injections: {len(injection_sample)}")

    print(f"number of injection DMs: {ndm}")
    print(f"number of injection FLUENCEs: {nfluence}")

    # header, freqs, rawdata,masked_data = get_filterbank_data_window(ifn, duration=duration,maskfn=maskfn)
    print(f"getting data cutout from: {ifn}")
    # add the pulses
    multiprocessing = True
    if multiprocessing:
        for dm in TRIAL_DMS:
            pool_arr = []
            for s in TRIAL_FLUENCE:
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
        rawdata, masked_data, masked_chans, presto_header = get_filterbank_data_window(
            ifn, duration=duration, maskfn=maskfn, masked_data=None
        )
        for dm in TRIAL_DMS:
            pool_arr = []
            for s in TRIAL_FLUENCE:
                pool_arr.append(
                    (
                        dm,
                        s,
                        ifn,
                        duration,
                        maskfn,
                        injection_sample,
                        rawdata,
                        masked_data,
                        masked_chans,
                        presto_header,
                        downsamp,
                        stats_window,
                        True,
                    )
                )
            for p in pool_arr:
                process(p)
