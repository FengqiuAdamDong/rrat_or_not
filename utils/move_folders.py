import numpy as np
import glob
import os
import shutil
if __name__ == '__main__':
    #find all the folders with inj_stats.dill
    path = glob.glob('./**/inj_stats.dill', recursive=True)
    #create a new directory called "pulsar_inj_stats"
    if not os.path.exists('pulsar_inj_stats'):
        os.makedirs('pulsar_inj_stats')
    for p in path:
        #remove the filename
        folder = p.replace('inj_stats.dill', '')
        positive_burst_file = folder + 'positive_bursts_1.csv'
        #get the folder name
        folder_name = folder.split('/')[-2]
        #make a new directory in pulsar_inj_stats
        if not os.path.exists('pulsar_inj_stats/' + folder_name):
            os.makedirs('pulsar_inj_stats/' + folder_name)
        #copy the inj_stats.dill file to the new directory
        shutil.copy(p, 'pulsar_inj_stats/' + folder_name + '/inj_stats.dill')
        #copy the positive_bursts_1.csv file to the new directory
        shutil.copy(positive_burst_file, 'pulsar_inj_stats/' + folder_name + '/positive_bursts_1.csv')
