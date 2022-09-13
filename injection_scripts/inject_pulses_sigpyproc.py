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
# define the random number generator
#np.random.seed(12345)

# TRIAL_SNR = [10]
    # 0.8,
# ]  # total S/N of pulse
# TRIAL_SNR = np.linspace(1,10,50)
TRIAL_SNR=[1,2,3,4,5,6,7,8,9,10]
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
    dtoa = np.zeros(npulses+1)
    while min(dtoa)<min_sep:
        #don't inject anything into the last 10% of data
        toas = sorted(np.random.uniform(0.1, 0.9, npulses) * duration)
        dtoa = np.diff(toas)
        if len(toas)==1:
            break

    # get the cartesian product so that we have a generator of TOA x SNR x DM
    grid_coords = np.array([*itertools.product(toas, TRIAL_SNR, TRIAL_DMS)])

    # save the injection sample to a numpy file
    print("saving injection sample to: sample_injections.npy")
    np.save("sample_injections.npy", grid_coords)

    return grid_coords, npulses, len(TRIAL_SNR), len(TRIAL_DMS)

def adjusted_peak(desired_a,tsamp,sigma,ds):
    #calculate the adjusted peak height after downsampling
    first_sum = int(ds/2)+1
    if ds%2==0:
        #even
        second_sum = int((ds-1)/2)
    else:
        #odd
        second_sum = first_sum
    first_term = 0
    second_term = 0
    for i in range(first_sum):
        first_term += np.exp(-i*tsamp**2/(2*sigma**2))
    for j in range(second_sum):
        i=j+1
        second_term += np.exp(-i*tsamp**2/(2*sigma**2))

    return ds*desired_a/(first_term+second_term)

def inject_pulses(data, masked_data,header, freqs, pulse_attrs,downsamp=4,plot=False):
    """For a given set of pulses, inject them into sample data"""
    # get the noise level in each channel
    # print("estimating initial noise values pre-injection")
    # per_chan_noise = np.std(data, axis=1)
    tsamp = header.tsamp
    statistics = []
    #10s stats window
    stats_window = 5
    for i, p in enumerate(pulse_attrs):
        p_toa, p_snr, p_dm = p
        print("computing toas per channel")
        # start the pulse 100 samples after the first simulated time step
        toa_bin_top = 0
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

        #get the data to do statistics
        stats_start,stats_end = (time_to_bin(p_toa-stats_window,tsamp),time_to_bin(p_toa+stats_window,tsamp))
        stats_nsamp = stats_end-stats_start
        stats_end = stats_end - (stats_nsamp%downsamp)
        print(f"injection TOA:{p_toa}")
        stats_data = copy.deepcopy(masked_data[:,stats_start:stats_end])
        # stats_data = copy.deepcopy(data[:,stats_start:stats_end])
        # stats_data = copy.deepcopy(data)
        end = stats_data.shape[1]-stats_data.shape[1]%4
        stats_data = stats_data[:,0:end]
        stats_data = stats_data.dedisperse(p_dm)
        ds_stats_data = stats_data.downsample(downsamp)
        #calculate off pulse std
        masked_mean = np.mean(ds_stats_data,0)
        masked_std = np.std(masked_mean)
        #calculate off pulse mean
        print("calculating expected S/N per channel")
        # convert S/N into actual power value
        width = 5e-3  # 5-ms FWHM pulses
        p_inj = adjusted_peak(p_snr,tsamp,width,downsamp)
        # p_inj = p_snr
        print(f"new peak snr {p_inj}")
        total_inj_pow = p_inj*masked_std
        print(f"masked_std: {masked_std}")
        # estimate per-channel power levels
        #we need to scale to uint8 later, so we add a fake number between -0.5 and 0.5 so that the average stays the same
        per_chan_inject_pow = total_inj_pow
        print(f"total power:{np.mean(per_chan_inject_pow)}")
        print("making pulses (Gaussians)")
        # simulate the pulse as a Gaussian, normalise such that the
        # peak corresponds to the per-channel power level corresponding
        # to the total S/N

        width_bins = time_to_bin(width, tsamp)
        pulse_wf = np.exp(
            -((x - per_chan_toa_bins[:, np.newaxis]) ** 2) / (2 * width_bins ** 2)
        )
        print("rescaling pulses to correct amplitude")
        pulse_wf /= pulse_wf.max(axis=1)[:, np.newaxis]
        pulse_wf *= per_chan_inject_pow
        pulse_wf += (np.random.rand(pulse_wf.shape[0],pulse_wf.shape[1])-0.5)

        ##################this is a controversial line!
        pulse_wf[pulse_wf<0]=0
        pulse_wf = np.around(pulse_wf)
        #create a spectra object and test
        # pulse_spec = spec.Spectra(data.freqs,data.dt,pulse_wf)
        # pulse_spec.dedisperse(100)
        # pulse_mean = np.mean(pulse_spec.data,axis=0)
        print("combining simulated pulse with data")
        true_start_bin = time_to_bin(p_toa, tsamp)-toa_bin_top
        true_end_bin = true_start_bin + pulse_wf.shape[1]
        print(f"start bin = {true_start_bin}  end bin = {true_end_bin}")
        # from sigpyproc.block import FilterbankBlock
        # pulse_wf = FilterbankBlock(pulse_wf,header = data.header)
        # pulse_wf = pulse_wf.dedisperse(100)
        # pulse_wf = pulse_wf.downsample(downsamp)
        # ts = np.mean(pulse_wf,axis=0)
        # plt.plot(ts)
        # plt.show()
        # import pdb; pdb.set_trace()
        #we're going to inject to 0
        #set the bins around the pulse to be the mean
        # per_chan_mean = np.median(stats_data,axis=1)
        # for i,b in enumerate(per_chan_toa_bins):
            # bins_to_reset = width_bins
            # data.data[i,true_start_bin+b-bins_to_reset:true_start_bin+b+bins_to_reset] = per_chan_mean[i]
        data[:, true_start_bin:true_end_bin] += pulse_wf

        # data[:,:]=50
        # injected_data = data[:,true_start_bin-100:true_end_bin]
        # injected_data =injected_data.dedisperse(100)
        # injected_data =injected_data.downsample(4)
        # plt.imshow(injected_data.normalise())
        # plt.show()
    #replot after scaling
    data = data.astype("uint8")
    # np.save('data',data)
    # for i, p in enumerate(pulse_attrs):
        # statistics.append(calculate_SNR_wrapper([p,stats_window,tsamp,downsamp,masked_data]))
    return data,statistics

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
    d = new_d[:,pulse_bin-plot_bin:pulse_bin+plot_bin]
    d = d.dedisperse(dm=p_dm)
    d = d.downsample(downsamp)

    ts = d.mean(axis=0)
    print("calculating snr")
    d_snr,a,s = calculate_SNR([ts,tsamp*downsamp,1e-2,int(plot_bin/downsamp)])
    print(f"Inj snr:{p_snr} Det snr: {d_snr} Amplitude:{a} std:{s}")
    return d_snr,a,s,p_toa,p_snr,p_dm

def calculate_SNR(X):
    #calculates the SNR given a timeseries
    ts = X[0]
    tsamp = X[1]
    width = X[2]
    nsamp = X[3]
    ind_max = nsamp
    w_bin = width/tsamp
    try:
        ts_std = np.delete(ts,range(int(ind_max-w_bin),int(ind_max+w_bin)))
        ts_std = ts
    except:
        print("ENCOUNTERED ERROR ***")
    # ts_std = ts
    mean = np.median(ts_std)
    std = np.std(ts_std)
    #subtract the mean
    ts_sub = ts-mean
    #remove rms
    Amplitude = ts_sub[nsamp]
    snr = Amplitude/std
    # print(np.mean(ts_sub))
    # plt.plot(ts_std)
    # plt.show()
    #area of pulse, the noise, if gaussian should sum, to 0
    return snr,Amplitude,std

def maskfile(maskfn, data, start_bin, nbinsextra):
    from presto import rfifind
    print('loading mask')
    rfimask = rfifind.rfifind(maskfn)
    print('getting mask')
    mask = get_mask(rfimask, start_bin, nbinsextra)[::-1]
    print('get mask finished')
    masked_chans = mask.all(axis=1)
    #mask the data but set to the mean of the channel
    mask_vals = np.median(data,axis=1)
    for i in range(len(mask_vals)):
        _ = data[i,:]
        _m = mask[i,:]
        _[_m] = mask_vals[i]
        data[i,:] = _
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



def get_filterbank_data_window(fn,maskfn, duration=20,masked_data = 1):
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
    #start in the middle of the data
    start = fil_dur / 2 - duration / 2
    stop = start + duration
    print("start stop bins",start,stop)
    start_bin = int(np.round(start / tsamp))
    stop_bin = int(np.round(stop / tsamp))
    nsamp = stop_bin-start_bin
    # get the data
    _ = filf.read_block(start_bin,nsamp)
    #read the block
    if masked_data == None:
        print("removing masked_channels")
        masked_data, masked_chans = maskfile(maskfn, copy.deepcopy(_), start_bin, nsamp)
        masked_data = masked_data[~masked_chans,:]

    else:
        masked_data, masked_chans = maskfile(maskfn, copy.deepcopy(_), start_bin, nsamp)
    freqs = FilterbankFile(fn).freqs
    header = FilterbankFile(fn).header
    # update the header so that it represents the windowed data
    hdr.tstart += (start_bin*tsamp) / 86400.0  # add in MJD
    header["tstart"] += (start_bin*tsamp) / 86400.0  # add in MJD
    hdr.nsamples = masked_data.shape[1]  # total number of data samples (nchan * nspec)
    return hdr, freqs, _, masked_data,header

def process(pool_arr):
    dm,snr,ifn,duration,maskfn,injection_sample,header_,freq_,rawdata_,masked_data_,header_presto = pool_arr
    header = copy.deepcopy(header_)
    freq = copy.deepcopy(freq_)
    rawdata = copy.deepcopy(rawdata_)
    masked_data = copy.deepcopy(masked_data_)
    print(f"dm={dm} snr={snr}")
    #figure out what pulses to add
    add_mask = np.logical_and(
        injection_sample[:, -1] == dm, injection_sample[:, 1] == snr
    )

    pulses_to_add = injection_sample[add_mask]


    injdata,statistics = inject_pulses(
        rawdata,
        masked_data,
        header,
        freqs,
        pulses_to_add,
    )
    s = np.array(statistics)
    # np.save(f"injection_stats_{snr}_{dm}",statistics)
    # we've done everything with type float32, rescale to uint8
    header_presto["nbits"] = 32

    ofn = os.path.basename(ifn).replace(".fil", f"_inj_dm{dm}_snr{snr}.fil")
    print(f"creating output file: {ofn}")
    # plt.plot(np.mean(injdata,axis=0))
    # plt.show()
    # NOTE: the filterbank spectra need to be provided with shape (nspec x nchan),
    # so we have to transpose the injected array at write time.
    # import pdb; pdb.set_trace()
    create_filterbank_file(ofn,header_presto,nbits=header_presto["nbits"],spectra=injdata.T)
    # injdata.to_file(ofn)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("fil", help="Filterbank file to use as raw data background")
    parser.add_argument("--m", help="this is the mask fn")
    parser.add_argument("--d", help="Duration of required output file", type=float, default=300)
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

    # header, freqs, rawdata,masked_data = get_filterbank_data_window(ifn, duration=duration,maskfn=maskfn)
    print(f"getting data cutout from: {ifn}")
    #add the pulses
    header, freqs, rawdata,masked_data,header_presto = get_filterbank_data_window(ifn, duration=duration,maskfn=maskfn,masked_data=None)
    for dm in TRIAL_DMS:
        pool_arr = []
        for s in TRIAL_SNR:
            pool_arr.append((dm,s,ifn,duration,maskfn,injection_sample,header, freqs, rawdata,masked_data,header_presto))
        for p in pool_arr:
            process(p)
        #with Pool(5) as p:
        #    p.map(process,pool_arr)
