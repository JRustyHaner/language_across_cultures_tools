# This script takes a frame from a video that represents the highest value of a given iMotions metric and saves it as a .jpg file.
# This script reads the data from a .csv file that contains the iMotions data and the video file.
# If there are two videos matching the name, the script will use the smallest in size.

import os
import csv
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from datetime import datetime



#paths
base_directory = '/media/rusty/Data2/UNGA/UNGA_78'
wmv_directory = os.path.join(base_directory, 'wmv')
mp3_directory = os.path.join(base_directory, 'mp3')
csv_directory = os.path.join(base_directory, 'csv_imotions')
output_directory = os.path.join(base_directory, 'jpg_stills_imotions')

#iterate through the csv files, and get the highest values
for filename in os.listdir(csv_directory):
    if filename.endswith('.csv'):
        csv_file_path = os.path.join(csv_directory, filename)
        print(f'Processing {csv_file_path}')
        #read the csv file
        df = pd.read_csv(csv_file_path, low_memory=False)
        #get the timestamp column of the highest value of each emotion: Anger,Contempt,Disgust,Fear,Joy,Sadness,Surprise,Engagement,Valence,Sentimentality,Confusion,Neutral
        #remove all columns that are not emotions except for timestamp
        try:
            df = df[['Timestamp','Anger','Contempt','Disgust','Fear','Joy','Sadness','Surprise','Engagement','Valence','Sentimentality','Confusion','Neutral']]
        except:
            print(f'Error processing {csv_file_path}')
            continue
        #print the number of rows
        print(f'Number of rows: {len(df)}')
        #iterate through the columns and get the top 3 highest values that aren't within 30 seconds of each other
        for column in df.columns:
            if column != 'Timestamp':
                print(f'Processing {column}')
                #get the top 3 highest values for each emotion
                highest_values = df.nlargest(1, column)
                #remove the indeces that are within 30 seconds of the highest value
                for index, row in highest_values.iterrows():
                    print(f'Processing {index}')
                    #get the timestamp of the highest value
                    timestamp = row['Timestamp']
                    #get the rows that are within 30 seconds of the highest value
                    rows_within_30_seconds = df[(df['Timestamp'] > timestamp - 30000) & (df['Timestamp'] < timestamp + 30000)]
                    #remove the highest value from the rows
                    rows_within_30_seconds = rows_within_30_seconds[rows_within_30_seconds['Timestamp'] != timestamp]
                    #remove the highest value from the dataframe
                    df = df[df['Timestamp'] != timestamp]
                    #remove the rows that are within 30 seconds of the highest value
                    df = df[~df['Timestamp'].isin(rows_within_30_seconds['Timestamp'])]
                #get the second highest value
                second_highest_values = df.nlargest(1, column)
                #remove the indeces that are within 30 seconds of the highest value
                for index, row in second_highest_values.iterrows():
                    print(f'Processing {index}')
                    #get the timestamp of the highest value
                    timestamp = row['Timestamp']
                    #get the rows that are within 30 seconds of the highest value
                    rows_within_30_seconds = df[(df['Timestamp'] > timestamp - 30000) & (df['Timestamp'] < timestamp + 30000)]
                    #remove the highest value from the rows
                    rows_within_30_seconds = rows_within_30_seconds[rows_within_30_seconds['Timestamp'] != timestamp]
                    #remove the highest value from the dataframe
                    df = df[df['Timestamp'] != timestamp]
                    #remove the rows that are within 30 seconds of the highest value
                    df = df[~df['Timestamp'].isin(rows_within_30_seconds['Timestamp'])]
                #get the third highest value
                third_highest_values = df.nlargest(1, column)
                #combine the highest values
                highest_values = pd.concat([highest_values, second_highest_values, third_highest_values])
                #drop any values that are 0
                highest_values = highest_values[highest_values[column] != 0]
                #print the top 3 highest values
                print(highest_values)
                #get the timestamp of the highest values
                highest_timestamps = highest_values['Timestamp']
                #print the timestamps
                print(highest_timestamps)
                #iterate through the video files, and match the first three letters of the filename with the csv file
                smallest_video_file_size = 0
                smallest_video_file_path = ''
                for video_filename in os.listdir(wmv_directory):
                    #if there are more than one video file that matches the filename, use the smallest in size
                    if video_filename.startswith(filename[:3]):
                        #get the video file path
                        video_file_path = os.path.join(wmv_directory, video_filename)
                        #get the video file size
                        video_file_size = os.path.getsize(video_file_path)
                        #if the video file size is less than the previous video file size, use the video file
                        if 'video_file_size' not in locals() or video_file_size < smallest_video_file_size:
                            smallest_video_file_size = video_file_size
                            smallest_video_file_path = video_file_path
                        print(f'Video file path: {smallest_video_file_path}')
                        video_file_path = os.path.join(wmv_directory, video_filename)
                        print(f'Processing {smallest_video_file_path}')
                        #get a still image from the video file of the top 3 highest values
                        cap = cv2.VideoCapture(video_file_path)
                        #get the frame rate
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        #get the number of frames
                        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                        #get the duration of the video
                        duration = frame_count/fps
                        #get the rank of each timestamp
                        #get the highest value
                        for timestamp in highest_timestamps:
                            #get the frame number
                            frame_number = int(timestamp/1000*fps)
                            #set the frame number
                            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                            #read the frame
                            ret, frame = cap.read()
                            #save the frame as a .jpg file
                            output_file_path = os.path.join(output_directory, f'{filename}_{column}_{timestamp}.jpg')
                            cv2.imwrite(output_file_path, frame)
                            print(f'Saved {output_file_path}')