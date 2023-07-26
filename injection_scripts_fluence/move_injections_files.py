import os
import numpy as np
import sys
import argparse
import glob
if __name__=='__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--path', type=str,default="injections",help='path to store the injections filterbank files (just a symlink)')
    argparser.add_argument('--gap', type=int,help='gap between two filterbank files')

    args = argparser.parse_args()

    path = args.path
    gap = args.gap

    files = os.listdir("./")
    files = [f for f in files if f.endswith(".fil")]
    #get pwd
    pwd = os.getcwd()
    #go up one level so that we're creating everything in a directory above
    pwd = os.path.join(pwd,"../")


    #check if path exists
    if not os.path.exists(path):
        os.makedirs(os.path.join(pwd,path))

    #symbolic link every gap-th file to path
    for i in range(0,len(files),gap):
        active_file = files[i]
        os.symlink(os.path.join(pwd,active_file), os.path.join(pwd,os.path.join(path,active_file)))
        #basename
        basename = active_file.split(".")[0]
        #symbolic link the base folder
        # os.symlink(os.path.join(pwd,basename+'/'), os.path.join(pwd,os.path.join(path,basename)))
        #create the basename folder
        new_folder = os.path.join(pwd,os.path.join(path,basename))
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)
        #list all rfifind files in the original folder
        rfifind_files = glob.glob(os.path.join(os.getcwd(),os.path.join(basename,"*rfifind*")))
        #symbolic link all rfifind files to the new folder
        for rfifind_file in rfifind_files:
            #check if file exits
            if not os.path.exists(os.path.join(new_folder,os.path.basename(rfifind_file))):
                print("symlinking "+rfifind_file+" to "+new_folder)
                os.symlink(rfifind_file, new_folder)
