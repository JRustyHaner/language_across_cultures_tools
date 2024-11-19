import os
import sys
from moviepy.editor import VideoFileClip
import multiprocessing
import csv
import pandas as pd

# Specify the input folder containing the WMA files
input_folder = '/media/Disk2/UNGA/UNGA_78/mp4/'

# Specify the output folder for WMA and MP3 files
output_folder_wmv= '/media/Disk2/UNGA/UNGA_78/wmv/'
output_folder_mp3= '/media/Disk2/UNGA/UNGA_78/mp3/'

#start a pandas dataframe to keep track of the files that have been converted
df = pd.DataFrame(columns=['filename'])




def main_function(input_folder, output_folder_wmv):
    while True:
        for filename in os.listdir(input_folder):
            #read the dataframe from the csv file
            try:
                df = pd.read_csv('converted_files.csv')
            except:
                df = pd.DataFrame(columns=['filename'])
            #read the dataframe to see if the file has already been converted
            if filename in df['filename']:
                print(f'{filename} already exists')
                continue
            else:
                 # Write the filename to the dataframe and save it
                df = df.append({'filename': filename}, ignore_index=True)
                df.to_csv('converted_files.csv', index=False)
            # Check if the file is an MP4 file
            if filename.endswith('.mp4'):
                # Get the full path of the input file
                input_path = os.path.join(input_folder, filename)

                #output file names
                stripped_filename = filename.split('.')[0]
                output_wmv_file = os.path.join(output_folder_wmv, stripped_filename + '.wmv')
                
                #check if the output files already exist
                if os.path.exists(output_wmv_file):
                    print(f'{filename} already exists')
                    continue

                # Load the video file
                print(f'Converting {filename} to WMV and MP3: input_path: {input_path}')
                video = VideoFileClip(input_path)
                
                #check if the wmv file already exists
                if os.path.exists(output_wmv_file):
                    print(f'{filename} already exists')
                    continue
                else:
                    try:
                        # Export the video to WMV format
                        video.write_videofile(output_wmv_file, codec='wmv2')
                    except:
                        print(f'Error converting {filename} to WMV')
                        #write to a file so we can check later
                        with open('error_files.txt','a') as error_file:
                            error_file.write(f'{filename}\n')
                        continue
                    

                # Get the audio from the video
            
                video.close()
            

               

# main to set arguments
if __name__ == '__main__':
    # usage input_folder output_folder_wmv
    if len(sys.argv) != 3:
        print('Usage: python convert_mp4_to_wmv_and_audio.py input_folder output_folder_wmv')
        sys.exit(1)
    input_folder = sys.argv[1]
    output_folder_wmv = sys.argv[2]

    # Check if the input folder exists
    if not os.path.isdir(input_folder):
        print(f'Error: {input_folder} does not exist')
        sys.exit(1)

    # Check if the output folder exists
    if not os.path.isdir(output_folder_wmv):
        print(f'Error: {output_folder_wmv} does not exist')
        sys.exit(1)

    # Convert the MP4 files to WMV and audio
    main_function(input_folder, output_folder_wmv)
