#!/usr/bin/env python3

import csv
import numpy as np
import sys
def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text  # or whatever

fn = sys.argv[1]

filfiles = []
dms = []
toas = []
with open(fn,'r') as csvf:
    p_csv = csv.reader(csvf,delimiter=',')
    for row in p_csv:
        fp = row[0].split('/')
        for _ in fp:
            if 'pow' in _:
                filname = _
                filfiles.append(_+'.fil')
                break
        info_str = remove_prefix(fp[-1],filname+'_')
        info_str = info_str.split('_')
        toas.append(info_str[0])
        dms.append(info_str[4])


out_csv = fn.replace('.csv','_snr.csv')

with open(out_csv,'w') as csvf:
    writer = csv.writer(csvf,delimiter=',')
    for write_row in zip(filfiles,dms,toas):
        writer.writerow(write_row)
