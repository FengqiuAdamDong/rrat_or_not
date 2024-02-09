import numpy as np
from csv import reader
import os
if __name__ == "__main__":
    # load the pulsars.csv file
    #
    csv_file = "pulsars.csv"
    with open(csv_file, 'r') as read_obj:
        csv_reader = reader(read_obj, delimiter=',')
        pulsar_name = []
        dm = []
        period = []
        for row in csv_reader:
            pulsar_name.append(row[0])
            dm.append(row[1])
            period.append(row[2])
    current_script = os.path.dirname(os.path.realpath(__file__))
    filter_script_path = f"{current_script}/../utils/filter_similar_burst.py"
    create_positive_csv_edit_path = f"{current_script}/../utils/create_positive_csv_edit.py"
    batch_submit_job_path = f"{current_script}/batch_extract_snr.sh"
    # for each pulsar, check if filtering has been done
    current_directory = os.getcwd()
    for p,period,dm in zip(pulsar_name,dm,period):
        # load the pulsar file
        foldername = f"{current_directory}/{p}/fdp/"
        os.chdir(foldername)
        # check if the file filtered.csv exists
        process_filter = True
        process_extract = True
        if os.path.exists("filtered.csv"):
            #check if filtered_edt.csv exists
            if os.path.exists("filtered_edt.csv"):
                process_filter = False
        extracted_path = f"{p}.dill"
        if os.path.exists(extracted_path):
            process_extract = False

        if process_filter:
            #print command
            print(f"python {filter_script_path} -folder_path positive_bursts_1 -dm {dm}")
            print(f"python {create_positive_csv_edit_path} filtered.csv")

            # os.system(f"python {filter_script_path} -folder_path positive_bursts_1 -dm {dm}")
            # os.system(f"python {create_positive_csv_edit_path} filtered.csv")
        if process_extract:
            #print command
            print(f"sbatch {batch_submit_job_path} {dm} {p} filtered_edt.csv {period}")
            # os.system(f"sbatch {batch_submit_job_path} {dm} {p} filtered_edt.csv {period}")
