import os
import numpy as np
import sys
import argparse
if __name__=='__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--path', type=str)
    argparser.add_argument('--gap', type=int,help='gap between two filterbank files')

    args = argparser.parse_args()

    path = args.path
    gap = args.gap

    files = os.listdir("./")
    files = [f for f in files if f.endswith(".fil")]

    #check if path exists
    if not os.path.exists(path):
        os.makedirs(path)

    #symbolic link every gap-th file to path
    for i in range(0,len(files),gap):
        active_file = files[i]
        #get pwd
        pwd = os.getcwd()
        os.symlink(os.path.join(pwd,active_file), os.path.join(pwd,os.path.join(path,active_file)))
        #basename
        basename = active_file.split(".")[0]
        #symbolic link the base folder
        os.symlink(os.path.join(pwd,basename+'/'), os.path.join(pwd,os.path.join(path,basename)))
