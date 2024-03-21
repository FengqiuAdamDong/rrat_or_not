import numpy as np
import inject_pulses_sigpyproc as inject
import argparse
import os
import glob
import sys
if __name__ == "__main__":
    #find the path of all files with the name sample_injections.npz
    sample_injections_files = glob.glob("**/sample_injections.npz", recursive=True)
    #get cwd
    cwd = os.getcwd()
    for sample_injections_file in sample_injections_files:
        #change to the directory of the sample_injections_file
        sample_dir = os.path.dirname(sample_injections_file)
        os.chdir(sample_dir)
        #get all the fil files
        fil_files = glob.glob('*SNR*width*.fil')
        #parse the fil files for SNR and width
        inj_files_properties = []
        for f in fil_files:
            #get the file size
            file_size = os.path.getsize(f)
            #if the file is less than 1mb then it's empty
            if file_size < 1000000:
                #remove the file
                print(f"Removing empty file: {f} file size: {file_size} bytes")
                os.remove(f)
                continue
            sp_ = f.split("_")
            #find the snr str and width str
            for s in sp_:
                if "SNR" in s:
                    snr_str = s
                if "width" in s:
                    width_str = s
            snr_str = snr_str.strip(".fil").strip("SNR")
            width_str = width_str.strip(".fil").strip("width")
            snr = np.round(float(snr_str), 4)
            width = np.round(float(width_str), 5)
            inj_files_properties.append([snr, width])

        #load the grid of injections
        sample_injections = np.load("sample_injections.npz",allow_pickle=True)
        grid = sample_injections['grid']
        snrs = grid[:,1]
        widths = grid[:,3]
        #all the same dm anyway
        dms = grid[:,2]

        unqiue_snrs = np.unique(snrs)
        unqiue_widths = np.unique(widths)
        all_injection_properties = []
        for s in unqiue_snrs:
            for w in unqiue_widths:
                all_injection_properties.append([np.round(s, 4), np.round(w, 5)])
        #now loop through and compare to see what's missing
        missing_injections = []
        for a in all_injection_properties:
            if a in inj_files_properties:
                continue
            else:
                print("Missing injection: ", a)
                script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
                # command = "sbatch "
                s = a[0]
                w = a[1]
                #all the same DM
                dm = dms[0]
                sbatch_command = f"sbatch {script_directory}/inject_individual_snr_width.sh {s} {w} {dm} {script_directory}"
                print(sbatch_command)
                os.system(sbatch_command)
        #do a reverse search to see if there are any extra files
        for a in inj_files_properties:
            if a in all_injection_properties:
                continue
            else:
                print("Extra injection: ", a)
        #change back to the original directory
        os.chdir(cwd)