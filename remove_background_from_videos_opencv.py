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
index = pd.DataFrame(columns=["Video Name", "Processed Video Name", "Segment Number", "Action"])


# function to create the index file
def create_index_file(output_folder):
    logger.info(f"Creating index file in the output folder: {output_folder}")
    with open(os.path.join(output_folder, index_file), "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Video Name", "Processed Video Name"])
    logger.info(f"Index file created successfully in the output folder")

# function to write the processed video name, segment number and the path to the index file
def write_to_index_file(output_folder, video_name, processed_video_name, segment_number, action):
    global index
    #get one directory above the output folder
    logger.info(f"Writing to index file in the output folder: {output_folder}")
    #read the index file
    if os.path.exists(os.path.join(output_folder, index_file)):
        index = pd.read_csv(os.path.join(output_folder, index_file))
    new_pd = pd.DataFrame([[video_name, processed_video_name, segment_number, action]], columns=["Video Name", "Processed Video Name", "Segment Number", "Action"])
    index = pd.concat([index, new_pd], ignore_index=True)
    index_file_final = index_file + "_" + video_name + "_" + segment_number + ".csv"
    index.to_csv(os.path.join(output_folder, index_file), index=False)
    print(f"Written to index file in the output folder: {output_folder}")

        
                  
    logger.info(f"Written to index file successfully in the output folder")

# function to check if the segment has already been processed
def check_processed(output_folder, video_name, segment_number, frame_number):
    logger.info(f"Checking if the segment has already been processed...")
    if os.path.exists(os.path.join(output_folder, f"{video_name}_segment_{segment_number}_frame_{frame_number}.png")):
        return True
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
def extract_frames(video, temp_dir, segment_number, segment_duration=2, video_name="video"):
    print(f"Extracting frames from video...")
    logger.info(f"Extracting frames from video {video_name}, segment {segment_number}, duration {segment_duration} seconds")
    #read the video
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_number = 0
    while frame_number < fps * segment_duration:
        ret, frame = video.read()
        if not ret:
            break
        # Save the frame as a png file
        frame_path = os.path.join(temp_dir, f"{video_name}_segment_{segment_number}_frame_{frame_number}.png")
        cv2.imwrite(frame_path, frame)
        logger.info(f"Saved frame {frame_number} of segment {segment_number} of video {video_name} to {frame_path}")
        frame_number += 1

    video.release()

def process_video(temp_dir, segment_number, video_name):
    # Remove all backgrounds of png files in the temp_dir using rembg with multithreading
    logger.info(f"Removing background from frames using rembg with multithreading...")
    file_count_total = len([f for f in os.listdir(temp_dir) if f.endswith(".png")])
    
    def process_frame(file):
        print(f"Processing file #{file} out of {file_count_total}")
        #parse the video name, segment number and frame number from the file name
        video_name, segment_number, frame_number = file.split("_")
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
        try: 
            num_segments = int(total_frames / (fps * chunk_duration)) + 1
        except ZeroDivisionError:
            num_segments = 1
        logger.info(f"Processing video {video_name} with {total_frames} frames at {fps} fps")
        print(f"There are {num_segments} segments in the video {video_name}. Frames per segment: {fps * chunk_duration}")

        #for each segment, extract the frames, remove the background and save the processed frames
        for segment_number in range(num_segments):
            temp_dir = os.path.join(output_folder, "temp_" + video_name + "_segment_" + str(segment_number))
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            # Check if the segment has already been processed
            if check_processed(output_folder, video_name, segment_number, 0):
                print(f"Segment {segment_number} of video {video_name} has already been processed. Skipping...")
                continue
            # Extract frames from the video
            video = cv2.VideoCapture(video_path)
            video.set(cv2.CAP_PROP_POS_FRAMES, segment_number * fps * chunk_duration)
            #extract frames concurrently
            extract_frames(video, temp_dir, segment_number, chunk_duration, video_name)
            frame_paths = []
            #go through all the segment and get the paths of the frames
            for frame_number in range(int(fps * chunk_duration)):
                frame_path = os.path.join(temp_dir, f"{video_name}_segment_{segment_number}_frame_{frame_number}.png")
                frame_paths.append(frame_path)

            
            def remove_background_from_frame(path):
                frame = cv2.imread(path)
                frame = rembg.remove(frame)
                cv2.imwrite(path, frame)
                logger.info(f"Removed background from frame {path}")
                os.rename(path, path.replace("frame", "processed_frame"))
                return path
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                executor.map(remove_background_from_frame, frame_paths)
                


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
