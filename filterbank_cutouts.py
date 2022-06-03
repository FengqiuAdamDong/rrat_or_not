#!/usr/bin/env python3
import numpy as np
from presto.filterbank import FilterbankFile
from presto.filterbank import create_filterbank_file as cbf
from presto import filterbank as fb
from presto import rfifind
from inject_stats import maskfile
from inject_stats_det import read_positive_file
from inject_stats_det import combine_positives
def filterbank_cutout(gf,ts,te,mask_fn,dm):
    #load the filterbank file
    g = FilterbankFile(gf,mode='read')
    tsamp = float(g.header['tsamp'])
    nsamps = int((te-ts)/tsamp)
    ssamps = int(ts/tsamp)
    #sampels to burst
    nsamps_start_zoom = int(2.5/tsamp)
    nsamps_end_zoom = int(3.5/tsamp)
    #spectra grabbed
    spec = g.get_spectra(ssamps,nsamps)
    #load mask
    data, masked_chans = maskfile(mask_fn,spec,ssamps,nsamps)
    #spectra masked
    fil_name = gf.strip('.fil')+f"_{ts}_{te}_{dm}"
    # write_header
    g.header['tstart'] = g.header['tstart']+ts/60/60/24
    cbf(fil_name+'.fil',header=g.header,spectra=data.data.T)
    print('saved',fil_name)
    np.save(fil_name+'_masked_chan',masked_chans)

def get_mask_fn_s(filterbank):
    folder = filterbank.strip('.fil')
    mask = f"{folder}_rfifind.mask"
    return mask


def get_mask_fn(filfiles):
    #get the filenames of all the masks
    mask_fn = [get_mask_fn_s(f) for f in filfiles]
    print(mask_fn)
    return mask_fn

fn = 'real_pulses/positive_bursts_edit_snr.csv'
fn1 = 'real_pulses/positive_bursts_1_edit_snr.csv'
fn2 = 'real_pulses/positive_bursts_short_edit_snr.csv'
# fn = 'real_pulses/positive_burst_test.csv'
# fn1 = fn
# fn2 = fn

fil1,dm1,toa1 = read_positive_file(fn)
fil2,dm2,toa2 = read_positive_file(fn1)
fil3,dm3,toa3 = read_positive_file(fn2)
print(len(fil1),len(dm1),len(toa1))
fil1,dm1,toa1 = combine_positives(fil1,fil2,dm1,dm2,toa1,toa2)
print(len(fil1),len(dm1),len(toa1))
fil1,dm1,toa1 = combine_positives(fil1,fil3,dm1,dm3,toa1,toa3)
print(len(fil1),len(dm1),len(toa1))
mask1 = get_mask_fn(fil1)
for f,d,t,m in zip(fil1,dm1,toa1,mask1):
    filterbank_cutout(f,t-3,t+3,m,d)
