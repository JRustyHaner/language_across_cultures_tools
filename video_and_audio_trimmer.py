# Description: This script trims the beginning of a video file and its associated audio file
# by using the timecode from the video file while playing it with mplayer.

import os
import csv
import subprocess
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip


# Function to trim a WMV file and its associated MP3 file
def trim_files(wmv_file, mp3_file, start_time):
    output_wmv_file = wmv_file.replace('.wmv', '_trimmed.wmv')
    output_mp3_file = mp3_file.replace('.mp3', '_trimmed.mp3')

    # Use MoviePy to calculate the video duration
    video = VideoFileClip(wmv_file)
    video_duration = video.duration
    video.close()

    start_time = float(start_time)  # Convert start_time to a float
    end_time = start_time + video_duration

    ffmpeg_extract_subclip(wmv_file, start_time, end_time, targetname=output_wmv_file)
    ffmpeg_extract_subclip(mp3_file, start_time, end_time, targetname=output_mp3_file)

    return output_wmv_file, output_mp3_file


def get_mplayer_timecode(mp4_file_path):
    command = ['mplayer', mp4_file_path]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    
    timecode = 0
    #while the process is running, we search for the timecode in the output. 
    while process.poll() is None:
        output = process.stdout.readline()
        if output.startswith('A:'):
            #we split by spaces and get the 2nd item in the list
            timecode = output.split()[1]
            print(timecode)        

    return timecode



# Base directory
base_directory = '/media/rusty/Data2/UNGA/UNGA_78/'

# Subdirectories for WMV, MP3, and MP4 files
wmv_directory = os.path.join(base_directory, 'wmv')
mp3_directory = os.path.join(base_directory, 'mp3')

# Create a CSV file to store the timecode and video file names
csv_filename = 'video_times.csv'
log_filename = 'process_log.txt'


with open(csv_filename, 'a', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)

    # Loop through WMV files in the "wmv" directory
    for filename in os.listdir(wmv_directory):
        #check if the file has already exists already, if so, skip
        if filename.endswith('.wmv'):
            if filename.endswith('_trimmed.wmv'):
                print(f'{filename} already processed, skipping.')
                continue
            #check if a corresponding wmv file exists
            trimmed_filename = filename.replace('.wmv', '_trimmed.wmv')
            if os.path.exists(os.path.join(wmv_directory, trimmed_filename)):
                print(f'{filename} already processed, skipping.')
                continue

            wmv_file_path = os.path.join(wmv_directory, filename)
            mp3_file_path = os.path.join(mp3_directory, filename.replace('.wmv', '.mp3'))


            if os.path.exists(wmv_file_path):
                start_time = get_mplayer_timecode(wmv_file_path)
                trim_files(wmv_file_path, mp3_file_path, start_time)
                csv_writer.writerow([filename, start_time])