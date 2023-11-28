# Description: This script will move all files in a directory to a folder with the same name as the file extension.

import os
import shutil

# Specify the source directory where your files are located
source_directory = './'

# Ensure the source directory exists
if not os.path.exists(source_directory):
    print(f"Source directory '{source_directory}' does not exist.")
    exit(1)

# Iterate over all files in the source directory
for filename in os.listdir(source_directory):
    print("filename: ", filename, "(file number: ", os.listdir(source_directory).index(filename) + 1, " of ", len(os.listdir(source_directory)), ")")
    file_path = os.path.join(source_directory, filename)

    # Check if the item is a file (not a directory)
    if os.path.isfile(file_path):
        # Get the file extension
        _, file_extension = os.path.splitext(filename)

        # Create a folder with the same name as the extension
        destination_directory = os.path.join(source_directory, file_extension.lstrip('.').lower())
        os.makedirs(destination_directory, exist_ok=True)

        # Move the file to the destination folder
        shutil.move(file_path, os.path.join(destination_directory, filename))

print("Files have been moved to folders based on their extensions.")
