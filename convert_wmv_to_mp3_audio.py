# Purpose: Convert all .wmv files in a folder to .mp3 files

import os
import subprocess
from moviepy.editor import *

def WMV_to_MP3(file):
    global output_folder
    #convert file from wmv to mp3
    #check if the mp3 file already exists
    mp3_file = file.replace('.wmv', '.mp3')
    if os.path.exists(mp3_file):
        print(f'{mp3_file} already exists')
        return
    else:
        try:
            # Load the video file
            print(f'Converting {file} to MP3')
            video = VideoFileClip(file)
            # Get the audio from the video
            audio = video.audio
            # Export the audio to MP3 format in the mp3s folder
            filename = os.path.basename(file)
            audio.write_audiofile(os.path.join(output_folder, filename[:-4] + '.mp3'))
            
        except Exception as e:
            print(f'Error converting {file} to MP3: {e}')
            #write to a file so we can check later
            with open('error_files.txt','a') as error_file:
                error_file.write(f'{file}\n')
            return

input_folder = '/media/rusty/Data2/UNGA/UNGA_78/wmv'
output_folder = '/media/rusty/Data2/UNGA/UNGA_78/mp3'

#iterate over files in the input folder ending in .wmv and convert them to .mp3
for file in os.listdir(input_folder):
    if file.endswith('_trimmed.wmv'):
        #check if the corresponding mp3 file already exists
        mp3_file = file.replace('.wmv', '.mp3')
        if os.path.exists(os.path.join(output_folder, mp3_file)):
            print(f'{mp3_file} already exists')
            continue
        else:
            WMV_to_MP3(os.path.join(input_folder, file))
