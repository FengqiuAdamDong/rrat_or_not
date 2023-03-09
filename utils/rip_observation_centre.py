#!/usr/bin/env python3
from presto.filterbank import FilterbankFile, create_filterbank_file
import sigpyproc
from sigpyproc import utils as u
import numpy as np
import os
#THIS SCRIPT ACTS TO RIP OUT THE CENTRE OF A PULSAR OBSERVATION
def get_filterbank_data(fn, duration=100):
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

    freqs = FilterbankFile(fn).freqs
    header = FilterbankFile(fn).header
    # update the header so that it represents the windowed data
    hdr.tstart += (start_bin*tsamp) / 86400.0  # add in MJD
    header["tstart"] += (start_bin*tsamp) / 86400.0  # add in MJD
    return hdr, freqs, _, header




if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("fil", help="Filterbank file to use as raw data background")
    parser.add_argument("-d", help="Duration of required output file", type=float, default=100)

    args = parser.parse_args()
    fil = args.fil
    duration = args.d
    header,freqs,spectra,header_presto = get_filterbank_data(fil,duration)
    ofn = os.path.basename(fil).replace(".fil", f"_{duration}_ripped.fil")

    create_filterbank_file(ofn,header_presto,nbits=header_presto["nbits"],spectra=spectra.T)
