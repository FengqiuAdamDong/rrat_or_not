import numpy as np
import csv
import sys
import os
#this script takes the `positive_bursts.csv` file output with CHIPSPIPE and will create a new one based on the files you have deleted
fn = sys.argv[1]
foldern = fn.split('.')[0]

#load the csv
flist = os.listdir(foldern)

#read the csv file
with open(f"{foldern}_edit.csv",'w') as csvfn1:
        editor = csv.writer(csvfn1,delimiter = ',')
        for f in flist:
            with open(fn,'r') as csvfn2:
                reader = csv.reader(csvfn2,delimiter = ',')
                if 'png' in f:
                    #if the file is a picture get the file name
                    f = f.split('.png')[0]
                    for row in reader:
                        fp = row[0].split('/')
                        if f == fp[-1]:
                            editor.writerow(row)
                            break
