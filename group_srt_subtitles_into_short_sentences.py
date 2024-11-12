#Description: This script combines subtitle entries into one entry 
#if the length is less than MAX_ENTRY_LENGTH or if the start time 
#is less than 5 seconds after the end time of the previous entry

import os
import re
import datetime


#set the directory for the srt files
input_dir = "/media/rusty/Data2/UNGA/UNGA_78/srt"

MAX_ENTRY_LENGTH = 30 # Set your desired maximum entry length (adjust as needed)


#iterate through all files in the directory ending with .srt
for filename in os.listdir(input_dir):
    print("Starting " + filename)

    #read the file
    with open(os.path.join(input_dir, filename), "r") as f:
        file_contents = f.read()

    #split the file contents based off of blank lines
    entries = file_contents.split("\n\n")
    print("Number of entries: " + str(len(entries)))

    subtitle_string = ""
    subtitle_entry_number = 1
    subtitle_start_time = ""
    subtitle_end_time = ""
    last_stop_time = datetime.datetime.strptime("00:00:00,000", "%H:%M:%S,%f")

    # Iterate through each entry
    for i in range(len(entries)):
        entry = entries[i]
        print("last_stop_time: " + str(last_stop_time))

        print("processing entry no. " + str(i) + " of " + str(len(entries)))
        # Split the entry based on new lines
        lines = entry.split("\n")

        # The first line is the entry number
        entry_number = lines[0]

        # The second line is the start time --> end time. Times are in format 00:00:00,000 --> 00:00:00,000
        start_time = lines[1].split(" --> ")[0]
        end_time = lines[1].split(" --> ")[1]

        print("start_time: " + start_time)
        print("end_time: " + end_time)

        #convert the start and end times to datetime objects. 
        start_time_hours = int(start_time.split(":")[0])
        start_time_minutes = int(start_time.split(":")[1])
        start_time_seconds = float(start_time.split(":")[2].split(".")[0])
        #round the seconds to 0 decimal places
        start_time_seconds = int(round(start_time_seconds, 0))
        #milliseconds should be 0  
        start_time_milliseconds = 0
        print("start_time: " + str(start_time_hours) + ":" + str(start_time_minutes) + ":" + str(start_time_seconds) + "," + str(start_time_milliseconds))
        start_time = datetime.datetime.strptime(str(start_time_hours) + ":" + str(start_time_minutes) + ":" + str(start_time_seconds) + "," + str(start_time_milliseconds), "%H:%M:%S,%f")

        end_time_hours = int(end_time.split(":")[0])
        end_time_minutes = int(end_time.split(":")[1])
        end_time_seconds = float(end_time.split(":")[2].split(".")[0])
        #round the seconds to 0 decimal places
        end_time_seconds = int(round(end_time_seconds, 0))
        #milliseconds should be 0
        end_time_milliseconds = 0
        print("end_time: " + str(end_time_hours) + ":" + str(end_time_minutes) + ":" + str(end_time_seconds) + "," + str(end_time_milliseconds))
        end_time = datetime.datetime.strptime(str(end_time_hours) + ":" + str(end_time_minutes) + ":" + str(end_time_seconds) + "," + str(end_time_milliseconds), "%H:%M:%S,%f")
        
        #if the subtitle_start_time is empty, set it to the start time of the first entry
        if subtitle_start_time == "":
            subtitle_start_time = start_time

        # The 3rd line is the subtitle
        subtitle = lines[2]

        #the purpose is to combine the subtitle entries into one entry if the length is less than MAX_ENTRY_LENGTH or if the start time is less than 5 seconds after the end time of the previous entry
        if len(subtitle_string) < MAX_ENTRY_LENGTH or (start_time - last_stop_time).total_seconds() < 5:
            subtitle_string += subtitle + " "
            subtitle_end_time = end_time
        else:
            #write the previous entry to the file
            new_filename = filename.replace(".srt", "_combined.srt")
            with open(os.path.join(input_dir, new_filename), "a") as f:
                f.write(str(subtitle_entry_number) + "\n")
                #convert the times from datetime objects to strings
                subtitle_start_time = subtitle_start_time.strftime("%H:%M:%S,%f")
                subtitle_end_time = subtitle_end_time.strftime("%H:%M:%S,%f")
                #remove the last 3 characters from the end of the string
                subtitle_start_time = subtitle_start_time[:-3]
                subtitle_end_time = subtitle_end_time[:-3]
                f.write(str(subtitle_start_time) + " --> " + str(subtitle_end_time) + "\n")
                f.write(subtitle_string + "\n\n")

            #reset the subtitle string
            subtitle_string = subtitle + " "
            subtitle_entry_number += 1
            subtitle_start_time = start_time
            subtitle_end_time = end_time
            last_stop_time = end_time

