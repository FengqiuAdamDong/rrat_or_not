from inject_pulses_sigpyproc import maskfile
from inject_pulses_sigpyproc import dm_delay
from inject_pulses_sigpyproc import time_to_bin
from inject_pulses_sigpyproc import add_pulse_to_data
from sigpyproc.block import FilterbankBlock as fbb
from sigpyproc import readers as r
from matplotlib import pyplot as plt
import numpy as np
import os
import copy
import sys



def write_data(my_data, my_outfn, my_filewriter=None, outbits=8):
    """writes my data to file
    parameters
    data: sigpyproc3 thing
    outfn: str, path to output file
    outbits: int, bit depth of output file

    returns
    None
    """
    # TODO: check if outfn exists
    if os.path.exists(my_outfn):
        print("outfile already exists, appending onto the end")
        outfile_header = r.FilReader(my_outfn).header
        # make sure that the start of this file is exactly at the end of the last one
        tend = (
            outfile_header.tstart
            + (outfile_header.nsamples * outfile_header.tsamp) / 86400.0
        )
        tstart_current = my_data.header.tstart
        if tend != tstart_current:
            print(
                "Warning: the end of the last file is not the same as the start of this file"
            )
            print(f"Last file end: {tend} Current file start: {tstart_current}")
            sys.exit(1)
        # append the data and adjust the nsamples
        write_data = my_data.astype(np.uint8).transpose().ravel()
        my_filewriter.cwrite(write_data)
        return my_filewriter
    else:
        updates = {"nsamples": my_data.header.nsamples}
        # data.to_file(outfn)
        outfile_writer = my_data.header.prep_outfile(my_outfn, updates=updates)
        # cast the data as 8 bits
        write_data = my_data.astype(np.uint8).transpose().ravel()
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
    #
    from sigpyproc.block import FilterbankBlock as fbb
    print("getting filterbank data")
    filf = r.FilReader(fn)
    hdr = filf.header
    # get the data
    _ = filf.read_block(start_bin, nsamp)
    try:
        masked_data, masked_chans = maskfile(
            mask_fn, copy.deepcopy(_.data), start_bin, nsamp
        )
    except:
        print("you're using an old version of sigpyproc3, it's fine though")
        masked_data, masked_chans = maskfile(
            mask_fn, copy.deepcopy(_), start_bin, nsamp
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
    print("Calculating pulse times")
    print(f"Period is {period}")
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

def inject_pulsar(data, pulse_attributes, freqs,tsamp):
    """injects a pulsar into the data
    parameters
    data: numpy array, data to inject the pulsar into
    pulse_attributes: list of tuples, attributes of the pulses
    freqs: numpy array, frequencies of the data

    returns
    data: numpy array, data with the pulsar injected
    """
    # get the pulse attributes
    combined_data = data
    #check if combined_data is an array
    if not isinstance(combined_data, np.ndarray):
        combined_data = data

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
            print('Failed at adding pulse at toa',ptoa,f"with dm sweep {nbins_to_sim*tsamp} and total time {data.shape[1]*tsamp}")
    data = combined_data
    # plt.imshow(data,aspect="auto")
    # plt.show()
    return data

def gaussian_noise(data):
    """replaces the data with gaussian noise
    parameters
    data: sigpyproc3 thing

    returns
    data: sigpyproc3 thing
    """
    data = fbb(np.random.normal(20, 3, data.shape),header=data.header)

    return data

def simulate_orbit(mjd_epoch, observation_epoch, period, outfn):
    G = 6.67430e-11 #si units
    c = 299792458
    solar_mass = 1.989e30
    sagA_mass = 4.3e6 #solar masses
    pulsar_mass = 1.4 #solarmasses
    T = 1*86400*365 #orbital period
    e = 0 #eccentricity
    w = 0 #longitude of periastron
    i = np.pi/4 # inclination angle, assuming 45 degrees for simplicity
    # i = 0
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
    #store the parameters as a yaml file
    yaml_dict= {
        "period": period,
        "mjd_epoch": mjd_epoch,
        "observation_epoch": observation_epoch,
        "T": T/86400/365,
        "A_T": A_T,
        "e": e,
        "w": w,
        "i": i,
        "V_L": float(V_L),
        "a_p": a_p,
        "shifted_period": float(P_shift),
    }
    with open(f"{outfn}.yaml", "w") as f:
        import yaml
        yaml.dump(yaml_dict, f)

    print("Shift params")
    print(1+V_L/c)
    print(A_T)
    return P_shift

def write_pulsar(X):
    real = X["real"]
    gaus_noise = X["gaus_noise"]
    i = X["i"]
    dm = X["dm"]
    period = X["period"]
    #reference epoch
    epoch = X["epoch"]
    pulse_width = X["pulse_width"]
    pulse_snr = X["pulse_snr"]
    downsamp = X["downsamp"]
    orbit_sampling_rate = X["orbit_sampling_rate"]

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
    gulp_size = 240  # seconds
    gulp_size_bins = int(gulp_size / tsamp)
    if gulp_size_bins > nsamples:
        gulp_size_bins = nsamples
    current_nsamp = 0

    mjd_to_add = i * orbit_sampling_rate
    outfn = f"injected_pulsar_{i}"
    #observation epoch is the start time of the data plus the mjd to add plus half the duration of the data
    observation_epoch = filf_header.tstart + mjd_to_add + (total_time / 2) / 86400.0
    p_shifted = simulate_orbit(epoch,observation_epoch,period,outfn)
    print(p_shifted)
    pulse_times = np.array(calculate_pulse_times(p_shifted, filf_header, epoch))

    filewriter = None
    if gaus_noise:
        filf = r.FilReader(real)
        hdr = filf.header

        hdr.tstart = hdr.tstart + mjd_to_add
        print(f"New start time {hdr.tstart}")
        chans = hdr.nchans
        samples = hdr.nsamples
        #generate a random array of the same size as the data
        #generate a random number between 1 and 10
        # data_std = np.random.uniform(1, 10)
        data_std = 1
        data = np.random.normal(15, data_std, (chans, samples))
        pulse_attributes = np.array(
            list((ptoa, pulse_snr, dm, pulse_width) for ptoa in pulse_times)
        )
        data = fbb(data, header=hdr)
        print(data)
        data = inject_pulsar(data, pulse_attributes, freqs, tsamp)
        filewriter = write_data(data, outfn+'.fil', my_filewriter=filewriter)
        filewriter = None
        #create a sigpyproc3 object
    else:
        while current_nsamp < nsamples:
            data, masked_chans = read_block(real, maskfn, current_nsamp, gulp_size_bins)
            data.header.tstart = data.header.tstart + mjd_to_add
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
            data = inject_pulsar(data, pulse_attributes, freqs, tsamp)
            current_nsamp += gulp_size_bins
            print(f"{current_nsamp}/{nsamples} samples read")
            # check to see how many samples are left
            if nsamples - current_nsamp < gulp_size_bins:
                gulp_size_bins = nsamples - current_nsamp
            filewriter = write_data(data, outfn+'.fil', my_filewriter=filewriter)
    print("done")
    current_nsamp = 0
    gulp_size_bins = int(gulp_size / tsamp)
    filewriter = None

if __name__ == "__main__":
    period = 2.123
    dm = 1778
    epoch = 50000 # MJD
    pulse_width = 10e-3  # seconds
    pulse_snr = 0.3
    downsamp = 1
    orbit_sampling_rate = 45 #days

    import argparse

    args = argparse.ArgumentParser()
    args.add_argument(
        "--real", type=str, default=None, help="Path to the real data file"
    )
    args.add_argument(
        "--gaus_noise", default=False, action="store_true", help="Add gaussian noise to the data"
    )
    args.add_argument(
        "--samples", default=12, type=int, help="Number of samples of the orbit", required=False
    )
    args.add_argument(
        "--dm", default=dm, type=float, help="DM of the pulsar", required=False
    )
    args.add_argument(
        "--period", default=period, type=float, help="Period of the pulsar", required=False
    )
    args.add_argument(
        "--epoch", default=epoch, type=float, help="Epoch of the pulsar", required=False
    )
    args.add_argument(
        "--pulse_width", default=pulse_width, type=float, help="Pulse width of the pulsar", required=False
    )
    args.add_argument(
        "--pulse_snr", default=pulse_snr, type=float, help="Pulse SNR of the pulsar", required=False
    )

    parser = args.parse_args()
    real = parser.real
    gaus_noise = parser.gaus_noise
    samples = parser.samples
    dm = parser.dm
    period = parser.period
    epoch = parser.epoch
    pulse_width = parser.pulse_width
    pulse_snr = parser.pulse_snr

    X_arr = []
    for i in range(samples):
        X = {
            "real": real,
            "gaus_noise": gaus_noise,
            "i": i,
            "pulse_width": pulse_width,
            "pulse_snr": pulse_snr,
            "dm": dm,
            "period": period,
            "epoch": epoch,
            "downsamp": downsamp,
            "orbit_sampling_rate": orbit_sampling_rate,
        }
    #     X_arr.append(X)
    #     write_pulsar(X)
    from multiprocessing import Pool
    with Pool(5) as p:
        p.map(write_pulsar, X_arr)
