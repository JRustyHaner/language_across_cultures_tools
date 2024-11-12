import os
import sys
import subprocess
import datetime
import shutil
import re
import cv2
import numpy as np
import rembg
from time import sleep
from tqdm import tqdm
import resource
import gc
from concurrent.futures import ThreadPoolExecutor

# Set a memory limit (in bytes)
memory_limit = 2 * 1024 * 1024 * 1024  # 2 GB

# Set maximum number of workers (threads)
max_workers = 1

def set_memory_limit(limit):
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

def remove_frame_background(frame):
    frame = np.array(frame)
    result = rembg.remove(frame, alpha_matting=True, alpha_matting_foreground_threshold=240, alpha_matting_background_threshold=10, alpha_matting_erode_structure_size=6, alpha_matting_base_size=1000)
    alpha = result[:, :, 3]
    result[alpha < 255] = [0, 255, 0, 255]
    result = result[:, :, :3]
    return result

def remove_background_worker(args):
    frame, temp_dir, i, j, k = args
    square_height_start = square_height * j
    square_height_end = min(square_height * (j + 1), frame.shape[0])  # Ensure it doesn't exceed frame height

    square_width_start = square_width * k
    square_width_end = min(square_width * (k + 1), frame.shape[1])  # Ensure it doesn't exceed frame width

    # Check if the calculated coordinates are valid
    if square_height_start >= square_height_end or square_width_start >= square_width_end:
        return i, frame  # Skip processing if the coordinates are invalid

    square = frame[square_height_start:square_height_end, square_width_start:square_width_end]
    square = remove_frame_background(square)
    print("Removing background from frame " + str(i + 1) + " square " + str(j * square_height + k + 1))
    
    frame[square_height_start:square_height_end, square_width_start:square_width_end] = square
    return i, frame

def convert_to_wmv(input_file, output_file):
    subprocess.run(["ffmpeg", "-i", input_file, "-c:v", "wmv2", "-b:v", "1024k", output_file], check=True)
    return output_file

def remove_background(input_video_file, output_video_file, n=1):
    set_memory_limit(memory_limit)
    
    try:
        video = cv2.VideoCapture(input_video_file)
        temp_dir = "temp_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        os.mkdir(temp_dir)

        with ThreadPoolExecutor(max_workers=max_workers) as executor, tqdm(total=int(video.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Processing Frames", position=0, leave=True, unit="frames") as pbar:
            for i in range(int(video.get(cv2.CAP_PROP_FRAME_COUNT))):
                try: 
                    success, frame = video.read()
                except cv2.error as e:
                    print("Error: " + str(e))
                    print("Clearing cache and trying again...")
                    sleep(5)
                    gc.collect()
                    success, frame = video.read()

                if not success:
                    break  # Break if there are no more frames to read

                frame_copy = frame.copy()

                args_list = [(frame_copy, temp_dir, i, j, k) for j in range(n) for k in range(n)]
                executor.map(remove_background_worker, args_list)

                pbar.update(1)
                pbar.set_postfix(CurrentFrame=i + 1)

                # Release the frame after processing
                del frame_copy
                gc.collect()

            # Process the remaining frames
            args_list = [(frame.copy(), temp_dir, i, j, k) for j in range(n) for k in range(n)]
            executor.map(remove_background_worker, args_list)

            for i, frame in args_list:
                cv2.imwrite(temp_dir + f"/frame{i + 1}.png", frame)
                pbar.update(1)
                pbar.set_postfix(CurrentFrame=i + 1)
    finally:
        video.release()  # Release the video capture object
        print("Done!")

if __name__ == "__main__":
    try:
        if len(sys.argv) != 3:
            print("Usage: python remove_background_from_video.py input_video output_video")
            sys.exit()

        input_video_file = sys.argv[1]

        if not os.path.exists(input_video_file):
            print("Error: input video file does not exist")
            sys.exit()

        if not re.search(r"\.mp4$|\.wmv$", input_video_file):
            print("Error: input video file must be in mp4 or wmv format")
            sys.exit()

        if re.search(r"\.mp4$", input_video_file):
            now = datetime.datetime.now()
            temp_dir = "temp_" + now.strftime("%Y%m%d%H%M%S")
            os.mkdir(temp_dir)
            input_video_file = convert_to_wmv(input_video_file, f"{temp_dir}/input_video.wmv")

        square_width = 0
        square_height = 0

        remove_background(input_video_file, sys.argv[2])
    finally:
        # Clean up temp directory if it was created
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
