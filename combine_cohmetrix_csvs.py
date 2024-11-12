# Description: Python script to combine all the Coh-Metrics csv files into one csv file
#
# Usage: python combine_cohmetrix_csvs.py <input_folder> <output_file>

import os
import sys
import csv
import pandas as pd

def combine_cohmetrix_csvs(input_folder, output_file):
    #each csv for a cohmetrix csv is rotated 90 degrees, so we need to transpose it.
    #check if output_file exists
    if os.path.exists(output_file):
        #if it does, read it in as a dataframe
        df = pd.read_csv(output_file, index_col=0)
    else:
        #if it doesn't, create an empty dataframe
        df = pd.DataFrame()
    #iterate through all the csv files in input_folder, and append them to the dataframe, skipping any that are already in the dataframe or output_file
    for filename in os.listdir(input_folder):
        if filename.endswith('.csv'):
            #check if filename is already in the dataframe
            if filename in df.index:
                print(f'{filename} is already in the dataframe, skipping')
                continue
            #check if filename is already in the output_file
            if filename.endswith(output_file) or filename.endswith('output.csv'):
                print(f'{filename} is the output file, skipping')
                continue
            #if filename is not in the dataframe or output_file, append it to the dataframe, we have to rotate it 90 degrees first since row 1 is the column names, and row 2 is the values
            print(f'Appending {filename} to dataframe')
            #read the csv into a dataframe
            temp_df = pd.read_csv(os.path.join(input_folder, filename), header=None)
            #transpose the dataframe
            temp_df = temp_df.transpose()
            #set the column names to the values in row 1
            temp_df.columns = temp_df.iloc[0]
            #drop row 1
            temp_df = temp_df.drop(0)
            #set the index to the filename
            temp_df.index = [filename]
            #append the temp_df to df
            df = pd.concat([df, temp_df])
            print(f'Finished appending {filename} to dataframe, the dataframe now has {len(df)} rows')
    #drop any duplicate rows
    print('Dropping duplicate rows')
    df = df.drop_duplicates()
    #write the dataframe to output_file
    print(f'Writing dataframe to {output_file}')
    df.to_csv(output_file)
    print(f'Finished writing dataframe to {output_file}')

    

    

if __name__ == '__main__':
    #handle if not 1 or 2 arguments. one argument is optional
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print('Usage: python combine_cohmetrix_csvs.py <input_folder> <output_file | default:input_folder/output.csv>')
        sys.exit(1)
    input_folder = sys.argv[1]
    #if output_file is not specified, default to input_folder/output.csv
    if len(sys.argv) == 2:
        output_file = os.path.join(input_folder, 'output.csv')
        print(f'Output file not specified, defaulting to {output_file}')
    else:
        output_file = sys.argv[2]
    #handle if output_file already exists
    if os.path.exists(output_file):
        print(f'{output_file} already exists, we will append to it')
    #handle if input_folder doesn't exist
    if not os.path.exists(input_folder):
        print(f'Input folder {input_folder} does not exist')
        sys.exit(1)

    combine_cohmetrix_csvs(input_folder, output_file)