#!/usr/bin/env python3

#!/usr/bin/env python3

import numpy as np
import yaml
import argparse
import csv
from sigpyproc import readers as r
import os
import glob
import dill
import matplotlib.pyplot as plt

# creates the yaml file for pulsar

parser = argparse.ArgumentParser(description="Create yaml file for pulsar")
parser.add_argument("csv_file", type=str, help="csv file with pulsar data")

args = parser.parse_args()
csv_file = args.csv_file

# read the csv file
pulsar_name = []
pulsar_period = []
with open(csv_file, "r") as f:
    reader = csv.reader(f)
    for row in reader:
        pulsar_name.append(row[0])
        pulsar_period.append(float(row[2]))

for pulsar in pulsar_name:
    os.chdir(f"{pulsar}/fdp/")
    # create the yaml file
    yaml_file = f"{pulsar}/fdp/{pulsar}.yaml"
    # check if yaml file already exists
    # get the root dir of current python script
    root_dir = os.path.dirname(os.path.realpath(__file__))
    dill_file = f"{pulsar}.dill"
    command = f"{root_dir}/batch_submit_fit.sh {dill_file}"
    print(command)
    os.system(command)
    os.chdir("../../")
