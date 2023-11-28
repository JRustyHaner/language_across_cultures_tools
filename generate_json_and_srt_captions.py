#Description: This script will generate JSON and SRT files for all the MP3 files in the MP3 folder using the Gentle server.

import os
import subprocess
import json

mp3_folder = '/media/rusty/Data2/UNGA/UNGA_78/mp3'
txt_folder = '/media/rusty/Data2/UNGA/UNGA_78/txt'
json_folder = '/media/rusty/Data2/UNGA/UNGA_78/json'
srt_folder = '/media/rusty/Data2/UNGA/UNGA_78/srt'
gentle_server_url = 'http://localhost:32771/transcriptions?async=false'

def generate_srt_captions(json_file, output_file):
    try:
        with open(json_file, 'r') as json_data:
            data = json.load(json_data)
    except FileNotFoundError:
        print(f"FileNotFoundError: {json_file} does not exist, skipping")
        return

    captions = []
    try:
        for word in data['words']:
            print(word)
            if word['case'] == 'not-found-in-audio' or not word['start']:
                continue
            start_time = round(word['start'], 3)
            end_time = round(word['end'], 3)
            text = word.get('alignedWord', '')  # Use .get() to provide a default value if 'alignedWord' is missing

            caption = f"{len(captions) + 1}\n{start_time} --> {end_time}\n{text}\n"
            captions.append(caption)
    except KeyError as e:
        print(f"KeyError: {e}. Make sure your JSON structure matches your code.")
        exit()
    except Exception as e:
        print(f"An error occurred: {e}")
        exit()

    with open(output_file, 'w') as srt_file:
        srt_file.write("\n".join(captions))

# Ensure the JSON output folder exists
if not os.path.exists(json_folder):
    os.makedirs(json_folder)

# Ensure the SRT output folder exists
if not os.path.exists(srt_folder):
    os.makedirs(srt_folder)

def generate_json_from_gentle(mp3_path, txt_path, json_path):
    #we may have to clean the output path since Gentle doesn't like spaces in the file name
    json_path = json_path.replace(' ', '_')
    # Use curl to send the files to the local Gentle server
    command = f"curl -F 'audio=@{mp3_path}' -F 'transcript=@{txt_path}' '{gentle_server_url}' > {json_path}"
    print(command)
    subprocess.call(command, shell=True)

    print(f'Sent files for {name} to Gentle server and saved JSON output to {json_path}')

# List all the MP3 files in the MP3 folder
mp3_files = os.listdir(mp3_folder)

for mp3_file in mp3_files:
    #if the file doesn't end with _trimmed.mp3, skip it
    if not mp3_file.endswith('_trimmed.mp3'):
        continue

    # Extract the name (without '_trimmed.mp3')
    name = mp3_file.replace('_trimmed.mp3', '')

    #if the txt file doesn't exist, skip it
    txt_name = name.replace('_78', '')
    if not os.path.exists(os.path.join(txt_folder, f'{txt_name}_trimmed_trimmed.txt')):
        print(f'{name}__trimmed_trimmed.txt does not exist')
        continue

    # Create the file paths
    mp3_path = os.path.join(mp3_folder, mp3_file)
    #we remove _78 from the txt file name because the txt files don't have _78 in them
    text_name = name.replace('_78', '')
    txt_path = os.path.join(txt_folder, f'{text_name}_trimmed_trimmed.txt')
    
    json_path = os.path.join(json_folder, f'{name}_trimmed.json')
    srt_path = os.path.join(srt_folder, f'{name}_trimmed.srt')

    print(f'Processing {name}')

    # Check if the MP3 and TXT files exist before processing
    if not os.path.exists(mp3_path) or not os.path.exists(txt_path):
        print(f'{mp3_path} or {txt_path} does not exist')
        continue

    # Check if the JSON and SRT files already exist
    if os.path.exists(json_path) and os.path.exists(srt_path):
        print(f'{json_path} and {srt_path} already exist')
        continue

    # Generate the JSON file
    generate_json_from_gentle(mp3_path, txt_path, json_path)
    


