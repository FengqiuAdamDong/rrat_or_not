from inject_pulses_sigpyproc import get_filterbank_data_window
from inject_pulses_sigpyproc import maskfile
from inject_pulses_sigpyproc import dm_delay
from inject_pulses_sigpyproc import time_to_bin
from inject_pulses_sigpyproc import add_pulse_to_data
from sigpyproc import readers as r
from matplotlib import pyplot as plt
import numpy as np
import os
import copy
import sys

# TODO: barycentric correction
#


def write_data(data, outfn, filewriter=None, outbits=8):
    """writes my data to file
    parameters
    data: sigpyproc3 thing
    outfn: str, path to output file
    outbits: int, bit depth of output file

    returns
    None
    """
    # TODO: check if outfn exists
    if os.path.exists(outfn):
        print("outfile already exists, appending onto the end")
        outfile_header = r.FilReader(outfn).header
        # make sure that the start of this file is exactly at the end of the last one
        tend = (
            outfile_header.tstart
            + (outfile_header.nsamples * outfile_header.tsamp) / 86400.0
        )
        tstart_current = data.header.tstart
        if tend != tstart_current:
            print(
                "Warning: the end of the last file is not the same as the start of this file"
            )
            print(f"Last file end: {tend} Current file start: {tstart_current}")
            sys.exit(1)
        # append the data and adjust the nsamples
        write_data = data.data.astype(np.uint8).transpose().ravel()
        filewriter.cwrite(write_data)
        return filewriter
    else:
        updates = {"nsamples": data.header.nsamples}
        # data.to_file(outfn)
        outfile_writer = data.header.prep_outfile(outfn, updates=updates)
        # cast the data as 8 bits
        write_data = data.data.astype(np.uint8).transpose().ravel()
        outfile_writer.cwrite(write_data)
        return outfile_writer


def read_block(fn, mask_fn, start_bin, nsamp=20):
    """reads a block of data from a filterbank file
    parameters
    fn: str, path to filterbank file
    time_block_size: int, size of time block in seconds

    returns
    data: sigpyproc3 thing
    """
    # find the archive filename
    # load the weights
    print("getting filterbank data")
    filf = r.FilReader(fn)
    hdr = filf.header
    # get the data
    _ = filf.read_block(start_bin, nsamp)
    masked_data, masked_chans = maskfile(
        maskfn, copy.deepcopy(_.data), start_bin, nsamp
    )
    if sum(masked_chans) != len(masked_chans):
        masked_data[masked_chans] = np.median(masked_data[~masked_chans])
    _._data = masked_data
    # find any nans in masked_data
    if np.isnan(masked_data).any():
        import pdb

        pdb.set_trace()
    return _, masked_chans


def calculate_pulse_times(period, header, mjd_epoch):
    """calculates the times of the pulses
    parameters
    period: float, period of the pulsar in seconds
    header: sigpyproc3 header object
    mjd_epoch: float, MJD epoch of the data

    returns
    pulse_times: list of floats, times of the pulses in MJD
    """
    # TODO: THERE IS A SOMETHING FISHY ABOUT THE PULSE TIMES THAT"S GOING ON HERE THE NUMBERS ARE TOO NICE
    # get the start time of the data
    start_time = header.tstart
    # get the total time of the data
    total_time = header.nsamples * header.tsamp
    # get the number of pulses
    n_pulses = int(total_time / period)
    # get the first pulse time from mjd_epoch
    rotations_since_epoch = int((start_time - mjd_epoch) * 86400 / period)
    first_pulse_time = mjd_epoch + ((rotations_since_epoch+1) * period / 86400)
    first_pulse_time_since_start = first_pulse_time - start_time
    first_pulse_time_since_start = first_pulse_time_since_start * 86400
    print(f"First pulse time is {first_pulse_time_since_start}")
    pulse_times = [first_pulse_time_since_start + i * period for i in range(n_pulses)]
    return pulse_times

def inject_pulsar(data, pulse_attributes, freqs):
    """injects a pulsar into the data
    parameters
    data: numpy array, data to inject the pulsar into
    pulse_attributes: list of tuples, attributes of the pulses
    freqs: numpy array, frequencies of the data

    returns
    data: numpy array, data with the pulsar injected
    """
    # get the pulse attributes
    combined_data = data.data
    for pulse in pulse_attributes:
        ptoa, pulse_snr, dm, pulse_width = pulse
        width_bins = int(pulse_width / data.header.tsamp)
        # get the delay
        # start the pulse 100 samples after the first simulated time step
        toa_bin_top = 0
        # assume TOA is arrival time at top of band
        max_dm_delay = dm_delay(dm, max(freqs), min(freqs))
        # print(f"max dm delay {max_dm_delay}")
        max_dm_delay_bins = time_to_bin(max_dm_delay, tsamp)
        # print(f"dm delay across band = {max_dm_delay} s = {max_dm_delay_bins} bins")
        nbins_to_sim = max_dm_delay_bins + 2 * toa_bin_top
        # pulse peak time at each frequency
        dm_delays = dm_delay(dm, freqs[0], freqs)
        per_chan_toa_bins = toa_bin_top + time_to_bin(dm_delays, tsamp)
        # calculate required injection amplitude
        # grab 1 second of data
        stats_window = int(1 / data.header.tsamp)
        stats_data = combined_data[:, :stats_window]
        std = np.std(np.mean(stats_data, axis=0))
        total_inj_power = pulse_snr * std
        try:
            combined_data = add_pulse_to_data(
                combined_data,
                ptoa,
                nbins_to_sim,
                per_chan_toa_bins,
                width_bins,
                total_inj_power,
                data.header.tsamp,
                toa_bin_top,
            )
        except:
            print('Failed at adding pulse at toa',ptoa,f"with dm sweep {nbins_to_sim*tsamp} and total time {data.data.shape[1]*tsamp}")
    data._data = combined_data
    # plt.imshow(data.data,aspect="auto")
    # plt.show()
    return data

def gaussian_noise(data):
    """replaces the data with gaussian noise
    parameters
    data: sigpyproc3 thing

    returns
    data: sigpyproc3 thing
    """
    data._data = np.random.normal(15, 5, data.data.shape)
    return data

def simulate_orbit(mjd_epoch, observation_epoch, period):
    G = 6.67430e-11 #si units
    c = 299792458
    solar_mass = 1.989e30
    sagA_mass = 4.3e6 #solar masses
    pulsar_mass = 1.3 #solarmasses
    T = 1*86400*365 #orbital period
    e = 0 #eccentricity
    w = np.pi/4 #longitude of periastron
    i = np.pi/4 # inclination angle, assuming 45 degrees for simplicity
    # T = 1 * 86400*365 #1 year in seconds
    a3 = (G*(sagA_mass*solar_mass)/(4*np.pi**2))*T**2
    a_p = a3**(1/3)
    au = 1.496e11
    print(a_p/au,"AU")
    #get T from observation epoch and mjd_epoch
    T_days = T/86400
    omega_b = 2*np.pi/T
    #assuming circular orbit and sini
    #try to find the phase of the orbit
    A_T = 2*np.pi*((observation_epoch - mjd_epoch) % T_days)/T_days #This is the phase in radians
    # omega = 0
    #add an eccentricity component


    V_L = omega_b * np.sin(i) * a_p * (np.cos(w+A_T)+e*np.cos(w))/(1-e**2)   # velocity line of sight in m/s

    P_shift = period*(1+V_L/c)
    print("Shift params")
    print(1+V_L/c)
    print(A_T)
    return P_shift

if __name__ == "__main__":
    period = 2
    dm = 1778
    epoch = 50000 # MJD
    pulse_width = 10e-3  # seconds
    pulse_snr = 0.01
    downsamp = 1
    orbit_sampling_rate = 30 #days
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument(
        "--real", type=str, default=None, help="Path to the real data file"
    )
    args.add_argument(
        "--gaus_noise", default=False, action="store_true", help="Add gaussian noise to the data"
    )
    args.add_argument(
        "--samples", default=12, type=int, help="Number of samples of the orbit", required=True
    )

    parser = args.parse_args()
    real = parser.real
    gaus_noise = parser.gaus_noise
    samples = parser.samples
    maskfn = real.split(".")[0] + "_01234568_rfifind.mask"
    if real:
        if not os.path.exists(maskfn):
            maskfn = real.split(".")[0] + "_rfifind.mask"
            if not os.path.exists(maskfn):
                print("No mask file found")
                exit()
    # get the basic header data of the filterbank file
    filf_header = r.FilReader(real).header
    nsamples = filf_header.nsamples
    tsamp = filf_header.tsamp
    total_time = nsamples * tsamp
    fch1 = filf_header.fch1
    foff = filf_header.foff
    nchan = filf_header.nchans
    freqs = np.linspace(fch1, fch1 + foff * nchan, nchan)
    gulp_size = 20  # seconds
    gulp_size_bins = int(gulp_size / tsamp)
    current_nsamp = 0
    filewriter = None
    for i in range(samples):
        mjd_to_add = i * orbit_sampling_rate
        injections_start_time = filf_header.tstart + mjd_to_add
        p_shifted = simulate_orbit(epoch,injections_start_time,period)
        print(p_shifted)
        pulse_times = np.array(calculate_pulse_times(p_shifted, filf_header, epoch))
        while current_nsamp < nsamples:
            data, masked_chans = read_block(real, maskfn, current_nsamp, gulp_size_bins)
            data.header.tstart = data.header.tstart + mjd_to_add
            if gaus_noise:
                data = gaussian_noise(data)

            current_time = current_nsamp * tsamp
            end_time = current_time + gulp_size_bins * tsamp
            # find the pulses in this gulp
            arg_pulses = (pulse_times > current_time) & (pulse_times < end_time)
            pulses = pulse_times[arg_pulses] - current_time
            # form pulse attributes
            pulse_attributes = np.array(
                list((ptoa, pulse_snr, dm, pulse_width) for ptoa in pulses)
            )
            # inject the pulses
            data = inject_pulsar(data, pulse_attributes, freqs)
            current_nsamp += gulp_size_bins
            print(f"{current_nsamp}/{nsamples} samples read")
            # check to see how many samples are left
            if nsamples - current_nsamp < gulp_size_bins:
                gulp_size_bins = nsamples - current_nsamp
            outfn = f"injections_{i}.fil"
            filewriter = write_data(data, outfn, filewriter=filewriter)
        print("done")
        current_nsamp = 0
        gulp_size_bins = int(gulp_size / tsamp)
        filewriter = None
