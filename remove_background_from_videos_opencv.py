# This script removes the background from a folder videos using OpenCV and saves the processed videos to a new folder. The script uses the cv2.createBackgroundSubtractorMOG2 method to create a background subtractor object and then applies it to each frame of the video to remove the background. The processed videos are saved in the output folder with the same name as the original videos.
# usage: remove_background_from_videos_opencv.py input_folder output_folder
#
import os
import sys
import cv2
import csv
import numpy as np
import concurrent.futures
import logging
import torch
import onnxruntime as ort
import rembg
import pandas as pd

list_of_files = dict()
CHUNK_DURATION = 2  # Duration of each chunk in seconds

#configure the logging
logging.basicConfig(filename='remove_background_from_videos_opencv.log', level=logging.DEBUG)
logger = logging.getLogger(__name__)
index_file = "index.csv"
index = pd.DataFrame(columns=["Video Name", "Processed Video Name", "Segment Number"])


# function to create the index file
def create_index_file(output_folder):
    logger.info(f"Creating index file in the output folder: {output_folder}")
    with open(os.path.join(output_folder, index_file), "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Video Name", "Processed Video Name"])
    logger.info(f"Index file created successfully in the output folder")

# function to write the processed video name, segment number and the path to the index file
def write_to_index_file(output_folder, video_name, processed_video_name, segment_number):
    logger.info(f"Writing to index file in the output folder: {output_folder}")
    with open(os.path.join(output_folder, index_file), "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([video_name, processed_video_name, segment_number])
    logger.info(f"Written to index file successfully in the output folder")

# function to check if the segment has already been processed
def check_segment_processed(output_folder, video_name, segment_number):
    logger.info(f"Checking if the segment has already been processed...")
    if os.path.exists(os.path.join(output_folder, index_file)):
        with open(os.path.join(output_folder, index_file), "r") as file:
            reader = csv.reader(file)
            next(reader)  # Skip header row
            for row in reader:
                if row[0] == video_name and int(row[2]) == segment_number:
                    logger.info(f"Segment {segment_number} of video {video_name} has already been processed")
                    return True
    logger.info(f"Segment {segment_number} of video {video_name} has not been processed yet")
    return False




# function to open the directory and get the list of files
def get_files(input_folder):
    logger.info(f"Getting list of video files in the input folder: {input_folder}")
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".mp4") or file.endswith(".wmv"):
                # remove any that end in .wmv.wmv
                if file.endswith(".wmv.wmv"):
                    continue
                video_path = os.path.join(root, file)
                list_of_files[video_path] = file
    logger.info(f"Found {len(list_of_files)} video files in the input folder")

# extract frames concurrently
def extract_frames(video, temp_dir, segment_start=0, segment_duration=CHUNK_DURATION):
    logger.info(f"Extracting frames from video...")
    success, frame = video.read()
    count = 0

    while success and count < segment_duration * video.get(cv2.CAP_PROP_FPS):
        frame_path = os.path.join(temp_dir, f"frame_{count}.png")
        cv2.imwrite(frame_path, frame)
        count += 1
        #every 30 frames, we will print the progress
        if count % 30 == 0:
            logger.info(f"Extracted {count} frames out of {segment_duration * video.get(cv2.CAP_PROP_FPS)} frames")
        

    video.release()

def process_video(temp_dir):
    # Remove all backgrounds of png files in the temp_dir using rembg with multithreading
    logger.info(f"Removing background from frames using rembg with multithreading...")
    file_count_total = len([f for f in os.listdir(temp_dir) if f.endswith(".png")])
    
    def process_frame(file):
        print(f"Processing file #{file} out of {file_count_total}")
        if file.endswith(".png"):
            if not file.startswith("processed_"):
                input_file = os.path.join(temp_dir, file)
                output_file = os.path.join(temp_dir, f"processed_{file}")
                with open(input_file, "rb") as input_image, open(output_file, "wb") as output_image:
                    output_image.write(rembg.remove(input_image.read()))
                # Delete the original file
                os.remove(input_file)
            else:
                print(f"File {file} has already been processed. Skipping...")
            print(f"Processed file #{file} out of {file_count_total}")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        executor.map(process_frame, os.listdir(temp_dir))
    
    logger.info(f"Removed background from all frames in the temp_dir")


# get the mask of the moving objects using color segmentation
def get_mask_color_segmentation(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #we are setting the parameters for the color segmentation based off of skin color
    lower = (0, 48, 80)
    upper = (20, 255, 255)

    mask = cv2.inRange(hsv, lower, upper)

    return mask

# remove the background from the frame by combining the masks
def remove_background(frame, mask1, mask2):
    mask = cv2.bitwise_or(mask1, mask2)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    return result


#combine the processed frames into a video (wmv format)
def combine_frames(output_folder, output_video):
    print("Combining frames into a video...")
    frame_files = [f for f in os.listdir(output_folder) if f.endswith(".png")]
    #files are processed_frame_#.png. Sort them based on the number
    frame_files.sort(key=lambda x: int(x.split("_")[2].split(".")[0]))

    frame = cv2.imread(os.path.join(output_folder, frame_files[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"WMV1")
    video = cv2.VideoWriter(output_video, fourcc, 30, (width, height))

    for frame_file in frame_files:
        frame = cv2.imread(os.path.join(output_folder, frame_file))
        video.write(frame)

    video.release()
    print(f"Combined {len(frame_files)} frames into {output_video}")

def get_mask_optical_flow(frame1, frame2):
    # Convert frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Calculate magnitude and angle of optical flow
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Threshold the magnitude to get the moving objects
    mask = cv2.threshold(magnitude, 10, 255, cv2.THRESH_BINARY)[1]

    return mask

# combine all images in an array of temp folders into a single video. 
# sort first by the segment number, then by the frame number
def combine_segment_videos(temp_dirs, output_video):
    print("Combining segment videos...")
    frames = []
    for temp_dir in temp_dirs:
        frame_files = [f for f in os.listdir(temp_dir) if f.endswith(".png")]
        frame_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
        for frame_file in frame_files:
            frame = cv2.imread(os.path.join(temp_dir, frame_file))
            frames.append(frame)

    height, width, layers = frames[0].shape

    fourcc = cv2.VideoWriter_fourcc(*"WMV1")
    video = cv2.VideoWriter(output_video, fourcc, 30, (width, height))

    for frame in frames:
        video.write(frame)

    video.release()
    print(f"Combined {len(frames)} frames into {output_video}")

def delete_temp_files(temp_dir):
    print("Deleting temporary files...")
    for file in os.listdir(temp_dir):
        file_path = os.path.join(temp_dir, file)
        os.remove(file_path)
    os.rmdir(temp_dir)
    print("Temporary files deleted")

def main(input_folder, output_folder, max_workers=4):
    get_files(input_folder)

    # Create the index file if it doesn't exist
    if not os.path.exists(os.path.join(output_folder, index_file)):
        create_index_file(output_folder)


    for video_path, video_name in list_of_files.items():
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        video.release()

        # Calculate the number of segments based on desired chunk duration
        chunk_duration = 5  # Adjust this as needed (seconds)
        num_segments = int(total_frames / (fps * chunk_duration)) + 1
        logger.info(f"Processing video {video_name} with {total_frames} frames at {fps} fps")
        print(f"There are {num_segments} segments in the video {video_name}. Frames per segment: {fps * chunk_duration}")

        for segment_number in range(num_segments):
            # Check if the segment has already been processed, if so, skip it
            if check_segment_processed(output_folder, video_name, segment_number):
                print(f"Segment {segment_number} of video {video_name} has already been processed. Skipping...")
                continue
            print(f"Processing segment {segment_number} of {num_segments} for video {video_name}...")
            segment_start = segment_number * chunk_duration * fps
            segment_end = min((segment_number + 1) * chunk_duration * fps, total_frames)

            video = cv2.VideoCapture(video_path)
            temp_dir = f"temp_{video_name}_{segment_number}"
            os.makedirs(temp_dir, exist_ok=True)

            # Extract frames for the current segment
            extract_frames(video, temp_dir, segment_start, chunk_duration)

            # Process frames to remove background
            process_video(temp_dir)

            # write to the index file
            processed_video_name = f"{video_name}_segment_{segment_number}.wmv"
            write_to_index_file(output_folder, video_name, processed_video_name, segment_number)

            video.release()

    # Function to process each video segment
    def process_segment(video_name):
        video_segments = index[index["Video Name"] == video_name]
        thisVideoTempDirs = []

        for i in range(len(video_segments) - 1):
            segment_number = video_segments.iloc[i]["Segment Number"]
            processed_video_name = video_segments.iloc[i]["Processed Video Name"]
            segment_path = os.path.join(output_folder, processed_video_name)
            thisVideoTempDirs.append(f"temp_{video_name}_{segment_number}")

        # Combine all segment videos into a single video
        output_video = os.path.join(output_folder, f"{video_name}_processed.wmv")
        combine_segment_videos(thisVideoTempDirs, output_video)

        # Delete temporary files
        for temp_dir in thisVideoTempDirs:
            delete_temp_files(temp_dir)

    # Use ThreadPoolExecutor to process video segments concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        futures = [executor.submit(process_segment, video_name) for video_name in list_of_files.values()]
        concurrent.futures.wait(futures)


    print("All videos processed successfully")

# check if the input arguments are provided, if not, print the usage.
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python remove_background_from_videos_opencv.py input_folder output_folder max_workers")
        sys.exit()

    #print the available ort providers
    available_providers = ort.get_available_providers()
    print(f"Available providers: {available_providers}")
    #check if CUDA is available
    if "CUDAExecutionProvider" in available_providers:
        print("CUDAExecutionProvider is available")
    else:
        print("CUDAExecutionProvider is not available")
        #check if the user wants to run anyways
        run_anyways = input("Do you want to run the script without CUDA? (y/n): ")
        if run_anyways.lower() != "y":
            sys.exit()

    #if the model folder exists, delete it (~/.u2net or C:\Users\username\.u2net)
    #windows delete C:\Users\username\.u2net
    #if os is windows
    if os.name == 'nt':
        model_folder = os.path.join("C:\\Users", os.getlogin(), ".u2net")
        if os.path.exists(model_folder):
            #it might have files in it. Delete the files first
            for file in os.listdir(model_folder):
                os.remove(os.path.join(model_folder, file))
            os.rmdir(model_folder)
    #if os is not windows
    else:
        model_folder = os.path.join(os.path.expanduser("~"), ".u2net")
        if os.path.exists(model_folder):
            os.rmdir(model_folder)

    #use np/cv2 to create a black image, then use rembg to remove the background
    print("Testing rembg...")
    black_image = np.zeros((512, 512, 3), dtype=np.uint8)
    cv2.imwrite("black_image.png", black_image)
    with open("black_image.png", "rb") as input_image, open("black_image_processed.png", "wb") as output_image:
        output_image.write(rembg.remove(input_image.read()))
    #delete the black image
    os.remove("black_image.png")
    os.remove("black_image_processed.png")
    print("Tested rembg successfully")


    

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    max_workers = int(sys.argv[3])

    main(input_folder, output_folder)
