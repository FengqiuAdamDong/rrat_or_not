#this will be a full injection pipeline
# 1. inject pulses
# 2. check that all the parameters have been injected
# 3. iteratively loop so that the maximum number of jobs at one time is 900
# 4. run check_single_pulse.py
# 5. iteratively loop so that the maximum number of jobs at one time is 900
# 6. run detect injected pulses
# 7. iteratively loop so that the maximum number of jobs at one time is 900


import os
import sys
import csv
import numpy as np


def run_inject_pulses(fil_files,dir_path):
    pulsar_dir = os.getcwd()
    for fil in fil_files:
        #remove the .fil extension
        fil_name = fil.split('.')[0]
        os.chdir(fil_name)
        mask_name = fil+'_rfifind.mask'
        command = f"python {dir_path}/inject_pulses_sigpyproc.py --m {mask_name} --d 150 --n 50 --sbatch --multi 1 {fil}"
        os.system(command)
        os.chdir(pulsar_dir)

if __name__=="__main__":
    #read in text file containing the list of pulsars
    pulsar_list = np.genfromtxt('pulsar_list.txt', dtype='str')
    dir_path = os.path.dirname(os.path.realpath(__file__))
    main_dir = os.getcwd()
    for pulsar in pulsar_list:
        #change to pulsar directory
        os.chdir(pulsar)
        fil_files = os.listdir()
        fil_files = [x for x in fil_files if x.endswith('.fil')]
