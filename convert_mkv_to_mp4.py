# Description: This script converts MKV files to MP4 files using moviepy.

import os
from moviepy.editor import VideoFileClip

# Define the input and output directories
input_dir = '/media/rusty/Data2/UNGA/UNGA_78/mkv'
output_dir = '/media/rusty/Data2/UNGA/UNGA_78/mp4'



files = os.listdir(input_dir)

# Loop through the files
for mkv_file in files:
    #remove files that are not mkv or have _video or _audio in the name
    if not mkv_file.endswith('.mkv') or '_video' in mkv_file or '_audio' in mkv_file:
        continue
    input_path = os.path.join(input_dir, mkv_file)
    output_file = os.path.splitext(mkv_file)[0] + '.mp4'
    output_path = os.path.join(output_dir, output_file)

    # Check if the output file already exists
    if os.path.exists(output_path):
        print(f'Skipping {mkv_file} - {output_file} already exists.')
        continue

    # Use moviepy to convert the MKV to MP4
    video = VideoFileClip(input_path)
    video.write_videofile(output_path, codec="libx264", audio_codec="aac")

    print(f'Converted {mkv_file} to {output_file}')

print('Conversion complete.')
