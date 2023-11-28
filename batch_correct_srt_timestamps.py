# Description: This script corrects the timestamp format in SRT files.

import os
import re

def correct_timestamp_format(srt_content):
    # Define a regular expression to match the timestamp format in your SRT file
    timestamp_pattern = re.compile(r'(\d+\.\d+) --> (\d+\.\d+)')

    # Replace the timestamp format with the correct one
    corrected_srt = re.sub(timestamp_pattern, lambda match: '{:02d}:{:02d}:{:06.3f},000'.format(
        int(float(match.group(1)) // 3600),  # hours
        int((float(match.group(1)) % 3600) // 60),  # minutes
        float(match.group(1)) % 60  # seconds including fractional part
    ) + ' --> ' + '{:02d}:{:02d}:{:06.3f},000'.format(
        int(float(match.group(2)) // 3600),  # hours
        int((float(match.group(2)) % 3600) // 60),  # minutes
        float(match.group(2)) % 60  # seconds including fractional part
    ), srt_content)

    return corrected_srt

def process_srt_file(file_path):
    with open(file_path, 'r') as file:
        srt_content = file.read()

    corrected_srt = correct_timestamp_format(srt_content)

    with open(file_path, 'w') as file:
        file.write(corrected_srt)

def batch_process_srt_files(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".srt"):
            file_path = os.path.join(folder_path, filename)
            process_srt_file(file_path)
            print(f"Processed: {file_path}")

# Replace 'your_folder_path' with the path to your folder containing SRT files
folder_path = '/media/rusty/Data2/UNGA/UNGA_78/srt'
batch_process_srt_files(folder_path)
