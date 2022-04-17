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
# define the random number generator
#np.random.seed(12345)

# TRIAL_SNR = [
#     1,
#     2,
#     3,
#     4,
#     5,
#     6,
#     7,
#     8,
# ]  # total S/N of pulse
TRIAL_SNR = np.linspace(0,10,200)
TRIAL_DMS = [
    100,
]  # DM of bursts

DM_CONST = 4149.377593360996  # dispersion constant


def dm_delay(dm, f1, f2):
    """Return DM delay in seconds"""
    return DM_CONST * dm * (1.0 / f2 ** 2 - 1.0 / f1 ** 2)


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

def create_pulse_attributes(npulses=1, duration=200,min_sep=1):
    """
    For a given number of pulses over a certain time duration,
    create an injection sample based on the pre-determined list
    of S/N and DM values.
    """
    dtoa = np.zeros(npulses-1)
    while min(dtoa)<min_sep:
        #don't inject anything into the last 10% of data
        toas = sorted(np.random.uniform(0.1, 0.9, npulses) * duration)
        dtoa = np.diff(toas)

    # get the cartesian product so that we have a generator of TOA x SNR x DM
    grid_coords = np.array([*itertools.product(toas, TRIAL_SNR, TRIAL_DMS)])

    # save the injection sample to a numpy file
    print("saving injection sample to: sample_injections.npy")
    np.save("sample_injections.npy", grid_coords)

    return grid_coords, npulses, len(TRIAL_SNR), len(TRIAL_DMS)

def inject_pulses(data, masked_data,header, freqs, pulse_attrs,plot=False):
    """For a given set of pulses, inject them into sample data"""
    # get the noise level in each channel
    # print("estimating initial noise values pre-injection")
    # per_chan_noise = np.std(data, axis=1)
    tsamp = header['tsamp']
    statistics = []
    for i, p in enumerate(pulse_attrs):
        p_toa, p_snr, p_dm = p
        print("computing toas per channel")
        # start the pulse 100 samples after the first simulated time step
        toa_bin_top = 100
        # assume TOA is arrival time at top of band
        max_dm_delay = dm_delay(p_dm, max(freqs), min(freqs))
        print(f"max dm delay {max_dm_delay}")
        max_dm_delay_bins = time_to_bin(max_dm_delay, tsamp)
        print(f"dm delay across band = {max_dm_delay} s = {max_dm_delay_bins} bins")
        nbins_to_sim = 2 * max_dm_delay_bins
        x = np.linspace(0, nbins_to_sim, nbins_to_sim)

        # pulse peak time at each frequency
        dm_delays = dm_delay(p_dm, freqs[0], freqs)
        per_chan_toa_bins = toa_bin_top + time_to_bin(dm_delays, tsamp)
        stats_start,stats_end = (time_to_bin(p_toa-max_dm_delay,tsamp),time_to_bin(p_toa+max_dm_delay,tsamp))
        print(f"injection TOA:{p_toa}")
        stats_data = masked_data.data[:,stats_start:stats_end]
        #calculate off pulse std
        masked_mean = np.mean(stats_data,0)
        masked_std = np.std(masked_mean)
        #calculate off pulse mean
        print("calculating expected S/N per channel")
        # convert S/N into actual power value
        total_inj_pow = p_snr*masked_std
        print(f"masked_std: {masked_std}")
        # estimate per-channel power levels
        #we need to scale to uint8 later, so we add a fake number between -0.5 and 0.5 so that the average stays the same
        per_chan_inject_pow = total_inj_pow
        print(f"total power:{np.mean(per_chan_inject_pow)}")
        print("making pulses (Gaussians)")
        # simulate the pulse as a Gaussian, normalise such that the
        # peak corresponds to the per-channel power level corresponding
        # to the total S/N
        width = 5e-3  # 2-ms FWHM pulses
        width_bins = time_to_bin(width, tsamp)
        pulse_wf = np.exp(
            -((x - per_chan_toa_bins[:, np.newaxis]) ** 2) / (2 * width_bins ** 2)
        )
        print("rescaling pulses to correct amplitude")
        pulse_wf /= pulse_wf.max(axis=1)[:, np.newaxis]
        pulse_wf *= per_chan_inject_pow
        pulse_wf += (np.random.rand(pulse_wf.shape[0],pulse_wf.shape[1])-0.5)
        pulse_wf = np.around(pulse_wf)
        print("combining simulated pulse with data")
        true_start_bin = time_to_bin(p_toa, tsamp) - toa_bin_top
        true_end_bin = true_start_bin + pulse_wf.shape[1]
        print(f"start bin = {true_start_bin}  end bin = {true_end_bin}")
        #we're going to inject to 0
        data.data[:, true_start_bin:true_end_bin] += pulse_wf
        statistics.append([masked_std,total_inj_pow])
        #plotting
        if False:
            new_d = copy.copy(data)
            new_d.dedisperse(dm=100)
            pulse_bin = int(p_toa/tsamp)
            d = new_d.data[:,pulse_bin-max_dm_delay_bins:pulse_bin+max_dm_delay_bins]
            ts = d.mean(axis=0)
            print(calculate_SNR(ts,tsamp,masked_std,10e-3))
            plt.plot(ts-ts.mean())
            plt.show()

    data.data = data.data.astype("uint8")

    return data,statistics

def calculate_SNR(ts,tsamp,std,width):
    #calculates the SNR given a timeseries

    ind_max = np.argwhere(np.max(ts)==ts)
    w_bin = width/tsamp
    ts_std = np.delete(ts,range(int(ind_max-w_bin),int(ind_max+w_bin)))
    # ts_std = ts
    mean = np.median(ts_std)
    # std = np.std(ts_std)
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

def maskfile(maskfn, data, start_bin, nbinsextra,extra_mask):
    from presto import rfifind
    print('loading mask')
    rfimask = rfifind.rfifind(maskfn)
    print('getting mask')
    mask = get_mask(rfimask, start_bin, nbinsextra)[::-1]
    print('get mask finished')
    masked_chans = mask.all(axis=1)
    # Mask data
    if extra_mask:
        masked_chans.append(extra_mask)
    print('masking')
    data = data.masked(mask, maskval='median-mid80')
    print('finished masking')
    print(np.sum(masked_chans))
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



def get_filterbank_data_window(fn,maskfn, duration=20,masked_data = None):
    """
    Open the filterbank file, extract the spectra around
    the middle (+/- duration / 2) and return that data as a
    2D array.
    """
    filf = FilterbankFile(fn)
    hdr = filf.header
    tsamp = hdr["tsamp"]
    fil_dur = filf.nspec * tsamp
    start = fil_dur / 2 - duration / 2
    stop = start + duration

    start_bin = int(np.round(start / tsamp))
    stop_bin = int(np.round(stop / tsamp))
    nsamp = stop_bin-start_bin
    # get the data
    _ = filf.get_spectra(start_bin, stop_bin - start_bin)
    if masked_data ==None:
        _copy = copy.copy(_)
        masked_data, masked_chans = maskfile(maskfn, _copy, start_bin, nsamp, False)
        # masked_data = masked_data.data.astype(np.float32)
        # data = _.data.astype(np.float32)
        masked_data.data = masked_data.data[~masked_chans,:]

    data = _
    freqs = _.freqs
    # update the header so that it represents the windowed data
    hdr["tstart"] += start / 86400.0  # add in MJD
    #hdr["nsamples"] = data.size  # total number of data samples (nchan * nspec)
    filf.close()

    return hdr, freqs, data, masked_data



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("fil", help="Filterbank file to use as raw data background")
    parser.add_argument("--m", help="this is the mask fn")
    parser.add_argument("--d", help="Duration of required output file", type=float, default=20)
    parser.add_argument("--n", help="Number of pulses to inject", type=int, default=2)
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
            default=None
    )

    args = parser.parse_args()

    duration = args.d
    ifn = args.fil
    maskfn = args.m
    if args.injection_file:
        injection_sample = np.load(args.injection_file)
        npul = len(injection_sample)
        ndm = len(set(injection_sample[:,2]))
        nsnr = len(set(injection_sample[:,1]))
    else:
        injection_sample, npul, nsnr, ndm = create_pulse_attributes(
            npulses=args.n, duration=duration
        )
    print(f"total number of injections: {len(injection_sample)}")

    print(f"number of injection DMs: {ndm}")
    print(f"number of injection SNRs: {nsnr}")

    header, freqs, rawdata,masked_data = get_filterbank_data_window(ifn, duration=duration,maskfn=maskfn)

    for dm in TRIAL_DMS:
        for snr in TRIAL_SNR:
            print(f"dm={dm} snr={snr}")
            #figure out what pulses to add
            add_mask = np.logical_and(
                injection_sample[:, -1] == dm, injection_sample[:, 1] == snr
            )

            pulses_to_add = injection_sample[add_mask]

            print(f"getting data cutout from: {ifn}")
            #add the pulses
            header, freqs, rawdata,__ = get_filterbank_data_window(ifn, duration=duration,maskfn=maskfn,masked_data=masked_data)

            injdata,statistics = inject_pulses(
                rawdata,
                masked_data,
                header,
                freqs,
                pulses_to_add,
            )
            np.save('injection_stats',statistics)
            # we've done everything with type float32, rescale to uint8
            # injdata = scale_to_uint8(injdata)
            header["nbits"] = 8

            ofn = os.path.basename(ifn).replace(".fil", f"_inj_dm{dm}_snr{snr}.fil")
            print(f"creating output file: {ofn}")
            # plt.plot(np.mean(injdata,axis=0))
            # plt.show()
            # NOTE: the filterbank spectra need to be provided with shape (nspec x nchan),
            # so we have to transpose the injected array at write time.
            fbout = create_filterbank_file(
                ofn, header, injdata.data.T, nbits=header["nbits"]
            )
            fbout.close()
