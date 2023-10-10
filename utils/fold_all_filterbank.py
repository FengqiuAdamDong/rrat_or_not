import numpy as np
import os
import sys
import argparse
from sigpyproc.readers import FilReader

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Fold all filterbank files in a directory')
    parser.add_argument('directory', type=str, help='Directory containing filterbank files')
    parser.add_argument('-par', '--parfile',default=None, type=str, help='Parfile to use for folding')
    parser.add_argument('-p',type=float, help='Period')
    args = parser.parse_args()
    all_files = os.listdir(args.directory)
    #if no par file is give, fold at args.p
    if args.parfile is None:
        for f in all_files:
            print(f)
            if f.endswith('.fil'):
                maskfn = f.replace('.fil','_rfifind.mask')
                print(f"folding with period {args.p}")
                os.system(f"prepfold -p {args.p} -mask {maskfn} -noxwin -nosearch -nsub 64 -npart 64 {f}")
    else:
        for f in all_files:
            if f.endswith('.fil'):
                maskfn = f.replace('.fil','_rfifind.mask')
                #figure out how many subints to use
                header = FilReader(f).header
                obs_time = header.tobs
                #read the par file
                with open(args.parfile, 'r') as parfile:
                    lines = parfile.readlines()
                for line in lines:
                    if "F0" in line:
                        f0 = float(line.split()[1])
                        p = 1/f0
                subints = int(obs_time/p)
                print(f"folding with subint {subints}")
                command = f"prepfold -mask {maskfn} -noxwin -nosearch -par {args.parfile} -nsub 128 -npart {subints} {f}"
                print(command)
                os.system(command)
                #run pazi on the folded files

    new_files = os.listdir(args.directory)
    for f in new_files:
        if f.endswith('.pfd'):
            os.system(f"pazi {f}")
