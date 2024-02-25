import numpy as np
from csv import reader
import os
import sys
import subprocess
import shutil
if __name__ == "__main__":
    # load the pulsars.csv file
    #
    #
    force_retry = True
    if force_retry:
        print("WARNING: Force retry is enabled, this will delete all the .dill and tmp.dill files in the pulsar folders.")
        input("Press Enter to continue...")
    csv_file = sys.argv[1]
    with open(csv_file, 'r') as read_obj:
        csv_reader = reader(read_obj, delimiter=',')
        pulsar_name = []
        dm = []
        period = []
        for row in csv_reader:
            if row[0][0] == "#":
                continue
            pulsar_name.append(row[0])
            dm.append(row[1])
            period.append(row[2])
    current_script = os.path.dirname(os.path.realpath(__file__))
    filter_script_path = f"{current_script}/../utils/filter_similar_bursts.py"
    create_positive_csv_edit_path = f"{current_script}/../utils/create_positive_csv_edit.py"
    batch_submit_job_path = f"{current_script}/batch_extract_snr.sh"
    # for each pulsar, check if filtering has been done
    current_directory = os.getcwd()
    for p,dm,period in zip(pulsar_name,dm,period):
        # load the pulsar file
        foldername = f"{current_directory}/{p}/fdp/"
        os.chdir(foldername)
        if force_retry:
            #if we force retry then remove the tmp.dill file and the pulsar.dill file
            if os.path.exists("tmp.dill"):
                os.remove("tmp.dill")
            if os.path.exists(f"{p}.dill"):
                os.remove(f"{p}.dill")
            if os.path.exists("fit_plots"):
                shutil.rmtree("fit_plots")

        # check if the file filtered.csv exists
        process_filter = True
        process_extract = True
        #check if filtered_edt.csv exists
        if os.path.exists("filtered_edit.csv"):
            process_filter = False
        extracted_path = f"{p}.dill"
        if os.path.exists(extracted_path):
            process_extract = False

        if process_filter:
            #print command
            print(f"python {filter_script_path} -folder_path positive_bursts_1 -dm {dm}")
            print(f"python {create_positive_csv_edit_path} filtered.csv")
            try:
                subprocess.run([f"python {filter_script_path} -folder_path positive_bursts_1 -dm {dm}"],check=True)
                subprocess.run([f"python {create_positive_csv_edit_path} filtered.csv"],check=True)
            except:
                continue
        if process_extract:
            #print command
            print(f"sbatch {batch_submit_job_path} {dm} {p} filtered_edit.csv {period}")
            os.system(f"sbatch {batch_submit_job_path} {dm} {p} filtered_edit.csv {period}")
        #go back to the original directory
        os.chdir(current_directory)
