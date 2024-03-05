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
import time

def run_inject_pulses(fil_files,dir_path):
    pulsar_dir = os.getcwd()
    for fil in fil_files:
        #remove the .fil extension
        fil_name = fil.split('.')[0]
        os.chdir(fil_name)
        #check if "sample_injections.npz" exists
        if os.path.exists('sample_injections.npz'):
            continue
        mask_name = fil+'_rfifind.mask'
        command = f"python {dir_path}/inject_pulses_sigpyproc.py --m {mask_name} --d 150 --n 50 --sbatch --multi 1 {fil}"
        print(command)
        os.system(command)
        os.chdir(pulsar_dir)

def run_check_inject_pulses(fil_files):
    fil_file_string = ''
    for fil in fil_files:
        fil_file_string += fil+' '
    command = f"python {dir_path}/inject_status_check.py {fil_file_string}"
    print(command)
    os.system(command)

def get_job_count_status(username="adamdong"):
    command = f"squeue -u {username} | wc -l"
    job_count = int(os.popen(command).read())
    return job_count

if __name__=="__main__":
    #read in text file containing the list of pulsars
    pulsars = np.genfromtxt('pulsar_list.txt', dtype='str')
    if pulsars.size == 1:
        pulsar_list = [pulsars]
    #convert all to str
    pulsar_list = [str(x) for x in pulsar_list]
    dir_path = os.path.dirname(os.path.realpath(__file__))
    main_dir = os.getcwd()
    for pulsar in pulsar_list:
        #change to pulsar directory
        os.chdir(pulsar)
        fil_files = os.listdir()
        fil_files = [x for x in fil_files if x.endswith('.fil')]
        #check if sample_injections.npz exists
        run_inject_pulses(fil_files,dir_path)
        #pause for 30 minutes
        time.sleep(1800)
        jobs_still_to_run = 2
        while jobs_still_to_run > 1:
            job_status_after_check = get_job_count_status()
            while job_status_after_check > 1:
                time.sleep(job_status_after_check*3)
                job_status_after_check = get_job_count_status()
            run_check_inject_pulses(fil_files)
            jobs_still_to_run = get_job_count_status()
