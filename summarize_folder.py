#This file lists all the filenames in a folder and counts number of files
import os

def make_filename_list(run_number):
    arr = sorted(os.listdir('runs//run_'+str(run_number)+'//EoS'))

    with open("runs//run_"+str(run_number)+"//filenames.txt","w") as file:
        file.write(str(len(arr))+"\n")
        for name in arr:
            file.write(name+"\n")
