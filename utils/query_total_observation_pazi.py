import psrchive
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sigpyproc.readers import FilReader

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Query total observation time and pazi')
    parser.add_argument('archive', nargs="+", help='archive file')
    args = parser.parse_args()
    total_observation_time = 0
    for archive in args.archive:
        arch = psrchive.Archive_load(archive)
        total_subint = arch.get_nsubint()
        total_chans = arch.get_nchan()
        total_bins = arch.get_nbin()
        mask_array = np.zeros((total_subint,1,total_chans,total_bins))
        for i,integration in enumerate(arch):
            for j in range(integration.get_nchan()):
                mask_array[i,0,j,:] = integration.get_weight(j)
        mask_array = np.mean(np.mean(np.mean(mask_array,axis=-1),axis=-1),axis=-1)
        #percentage masked
        mask_percentage = np.sum(mask_array==0)/total_subint
        #get the filterbank filename
        ar_split = archive.split('_')
        filterbank_filename = ''
        for i,ar in enumerate(ar_split):
            if ar == 'PSR':
                break
            elif 'ms' in ar:
                break
            if i == 0:
                filterbank_filename += ar
            else:
                filterbank_filename += '_'+ar

        filterbank_filename += '.fil'
        header = FilReader(filterbank_filename).header
        total_observation_time += header.tobs*(1-mask_percentage)
    print('Total observation time: {} s'.format(total_observation_time))
