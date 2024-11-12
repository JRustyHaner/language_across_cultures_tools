#Description: This script extracts a column from a csv file and writes it to a txt file
#usage: python extract_csv_column_to_txt.py input_file.csv column_number_for_extraction column_number_for_filename output_directory
#note, column nums start at 0

import os
import re
import sys
import datetime
import pandas as pd

#check if the correct number of arguments were passed to the script, otherwise print usage
if len(sys.argv) != 5:
    print("Usage: python extract_csv_column_to_txt.py input_file.csv column_number_for_extraction column_number_for_filename output_directory")
    sys.exit()

#get the input file by reading the argument passed to the script
input_file = sys.argv[1]

#set the column number to extract
column_number = int(sys.argv[2])

#set the column number to use for the filename
filename_column_number = int(sys.argv[3])

#set the output directory
output_dir = sys.argv[4]

#read the csv file into pandas
pd_file = pd.read_csv(input_file)

#iterate through the rows, extracting the column and writing it to a txt file
for index, row in pd_file.iterrows():
    #get the column value
    column_value = row[column_number]

    #remove any text after the copyright symbol, if it exists. Also remove the copyright symbol
    column_value = re.sub(r'©.*', '', column_value)
    column_value = column_value.replace("©", "")

    #get the filename
    filename = row[filename_column_number]

    #remove any punctuation from the filename, replace spaces with underscores
    filename = re.sub(r'[^\w\s]','',filename)
    filename = filename.replace(" ", "_")

    #append .txt to the filename
    filename = filename + ".txt"

    #write the column value to a txt file
    with open(os.path.join(output_dir, filename), "w") as f:
        f.write(column_value)

    print("Wrote " + filename + " to " + output_dir + " directory")
    

print("Done!")



