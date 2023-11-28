#This script is used to replace the audio in a video file with the audio from another video file.

import os
from moviepy.editor import VideoFileClip, AudioFileClip

def replace_audio(video_file, audio_file, output_file):
    video = VideoFileClip(video_file)
    audio = AudioFileClip(audio_file)
    video = video.set_audio(audio)
    video.write_videofile(output_file, codec='libx264')

def main():
    video_folder = '/media/rusty/Data2/UNGA/UNGA_78/mkv/'
    video_files = [f for f in os.listdir(video_folder) if f.endswith('_video.mkv')]

    for video_file in video_files:
        video_name = os.path.basename(video_file).replace('_video.mkv', '')
        audio_file = os.path.join(video_folder, f"{video_name}_audio.mkv")
        
        if os.path.exists(audio_file):
            output_file = os.path.join(video_folder, f"{video_name}_new.mkv")
            replace_audio(os.path.join(video_folder, video_file), audio_file, output_file)
            print(f"Replaced audio for {video_file} and saved as {output_file}")
        else:
            print(f"Audio file for {video_file} not found at {audio_file}")

if __name__ == "__main__":
    main()
