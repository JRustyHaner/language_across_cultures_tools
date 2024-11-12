#Description: This script superimposes captions on WMV video files using SRT files.

from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
import pysrt
import os

# Define input and output directories
video_folder = '/media/rusty/Data2/UNGA/UNGA_78/wmv'
srt_folder = '/media/rusty/Data2/UNGA/UNGA_78/srt'
output_folder = "/media/rusty/Data2/UNGA/UNGA_78/mp4"

# Iterate over video files in the input directory
for video_file in os.listdir(video_folder):
    # Check if the file is a WMV video file
    if video_file.endswith("trimmed.wmv"):
        # Formulate the full path to the video file
        video_path = os.path.join(video_folder, video_file)
        
        # Formulate the path to the corresponding SRT file
        srt_file = os.path.join(srt_folder, video_file.replace(".wmv", "_combined.srt"))

        # Check if the SRT file exists for the current video
        if not os.path.exists(srt_file):
            print(f"Skipping {video_file} as the SRT file does not exist.")
            continue

        # Formulate the output file path with captions
        output_file = os.path.join(output_folder, video_file.replace(".wmv", "_with_captions.mp4"))

        # Check if the output file already exists
        if os.path.exists(output_file):
            print(f"Skipping {video_file} as it already has an output file with captions.")
            continue

        # Read the content of the SRT file
        with open(srt_file, 'r') as srt:
            subtitles_content = srt.read()

        # Load the video file using MoviePy
        video = VideoFileClip(video_path)

        # Load the subtitles from the SRT file using pysrt
        subtitles = pysrt.open(srt_file)

        # Iterate over each subtitle to create TextClips for captions
        text_clips = []
        for subtitle in subtitles:
            print("progress: " + str(subtitle.start.seconds) + " of " + str(video.duration) + " seconds")
            text_clips.append(
                TextClip(subtitle.text, fontsize=24, color='white', font='Arial', method='caption', size=(video.w, None))
                .set_duration(subtitle.end.seconds - subtitle.start.seconds + subtitle.end.milliseconds / 1000)
            )

        # Create a CompositeVideoClip by overlaying TextClips on the original video
        video_with_captions = CompositeVideoClip([video] + text_clips)

        # Write the final video with captions to a file
        print("Writing " + output_file)
        video_with_captions.write_videofile(output_file, codec='libx264', audio_codec='aac')

# Notify the user that the captioning process has been completed
print("Captioning process completed.")
