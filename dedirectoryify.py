## script to scan the current directory and all subdirectories for files and move them to the current directory

import os
import shutil
import sys

def main():
    # get the current directory
    current_dir = os.getcwd()
    # get the list of files in the current directory
    files = os.listdir(current_dir)
    # loop through the files
    for file in files:
        # if the file is a directory
        if os.path.isdir(file):
            # get the list of files in the directory
            sub_files = os.listdir(file)
            # loop through the files in the directory
            for sub_file in sub_files:
                # move the file to the current directory
                shutil.move(os.path.join(current_dir, file, sub_file), os.path.join(current_dir, sub_file))
            # remove the directory
            os.rmdir(os.path.join(current_dir, file))
    # exit the program
    sys.exit()

if __name__ == '__main__':
    main()