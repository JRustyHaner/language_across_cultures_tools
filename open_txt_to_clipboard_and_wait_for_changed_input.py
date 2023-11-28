# Description: This script will open a txt file, 
# copy the contents to the clipboard, and wait for the user to edit the contents.
# Once the user presses enter, the script will save the edited contents to a new file. 
# The script will then open the next file in the folder and repeat the process.
# The script will save the last processed file and prompt number to a checkpoint file. 
# If the script is interrupted, it will recover from the last processed file and prompt number.

import os
import subprocess
import re

# Add these variables to track the last processed file and prompt
last_processed_file = ""
last_processed_prompt = 0

import os
import subprocess
import re

def main():
    global last_processed_file, last_processed_prompt  # Declare as global

    folder_path = "/media/rusty/Data2/UNGA/UNGA_78/txt"

    try:
        load_checkpoint()  # Load the last processed state
        for root, dirs, files in os.walk(folder_path):
            for filename in files:
                if filename == last_processed_file:
                    last_processed_file = ""  # Reset the last processed file
                    continue

                #check if the file name contains _trimmed, if so, skip
                if '_trimmed' in filename:
                    continue

                exists = False
                #iterate through all files, searching for files that contain the file name without the extension that also contains _trimmed
                #if the file exists, skip
                for file in files:
                    if filename[:-4] in file and '_trimmed' in file:
                        print(f'{filename} already exists')
                        exists = True

                if exists:
                    print(f'{filename} trimmed already exists')
                    continue

                #if the file ends in _trimmed.txt, skip
                if filename.endswith('_trimmed.txt'):
                    continue



                print(f"Processing {filename} (file number: {files.index(filename) + 1} of {len(files)})")

                last_processed_prompt = 0  # Initialize the prompt number for each file

                file_content = get_file_content(os.path.join(root, filename))
                file_path = os.path.join(root, filename.replace(".txt", "_trimmed.txt"))

                if len(file_content) > 0:
                    prompts = split_content_into_prompts(file_content)
                    for i in range(last_processed_prompt, len(prompts)):
                        prompts[i] = preprocess_prompt(prompts[i])
                        print(f"Prompt {i + 1} of {len(prompts)} for {filename}:")
                        copy_prompt_to_clipboard(prompts[i], filename, i + 1)
                        input("Press Enter when you are done editing the prompt: ")
                        user_input = get_from_clipboard()
                        prompts[i] = user_input
                        last_processed_prompt = i

                    print("Prompts before saving:")
                    print(prompts)

                    save_edited_content(file_path, prompts)

                    print("Prompts after saving:")
                    print(prompts)

                    # Update the checkpoint when processing is completed for this file
                    update_checkpoint(filename, len(prompts))

    except KeyboardInterrupt:
        print("\nOperation interrupted. The script will recover from the last processed file.")
    except Exception as e:
        print(f"Error: {str(e)}")

def get_file_content(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def split_content_into_prompts(file_content, max_prompt_length=5000):
    prompts = []
    while len(file_content) > max_prompt_length:
        index = file_content.rfind("\n", 0, max_prompt_length)
        if index == -1:
            index = max_prompt_length
        prompts.append(file_content[:index])
        file_content = file_content[index:]
    prompts.append(file_content)
    return prompts
    
def copy_prompt_to_clipboard(prompt, filename, prompt_number):
    prompt_text = f"Edit and press Enter for {filename} (Prompt {prompt_number}):\n{prompt}"
    prompt_prepend = "Please edit this to remove URLs, number lists, hypen lists, or page numbers. Please also remove new lines and paragraph returns:\n"
    
    # Write the prompt to the clipboard
    clipboard_content = prompt_prepend + prompt_text
    subprocess.run(['xclip', '-selection', 'clipboard'], input=clipboard_content, encoding='utf-8')

def get_from_clipboard():
    # Get the clipboard content
    clipboard_content = subprocess.run(['xclip', '-selection', 'clipboard', '-o'], stdout=subprocess.PIPE, encoding='utf-8').stdout
    #remove any paragraphs or new lines that end in a colon
    clipboard_content = re.sub(r':\n', '', clipboard_content)
    #remove any phrases that end in a colon, such as Introduction:
    clipboard_content = re.sub(r'[^\.]\n', '', clipboard_content)
    #remove any new lines from paragraph returns or new lines
    clipboard_content = clipboard_content.replace('\n',' ')
    clipboard_content = clipboard_content.replace('\r',' ')
    
    return clipboard_content

def preprocess_prompt(prompt):
    #remove lines that end in colon
    prompt = re.sub(r':\n', '', prompt)
    #remove any phrases that end in a colon, such as Introduction:
    prompt = re.sub(r'[^\.]\n', '', prompt)
    #we remove new lines and paragraph returns
    prompt = prompt.replace('\n',' ')
    prompt = prompt.replace('\r',' ')
    return prompt
    

def save_edited_content(file_path, prompts):
    new_file_path = os.path.splitext(file_path)[0] + "_trimmed.txt"
    with open(new_file_path, 'w') as file:
        file.write('\n'.join(prompts))
    print(f"Saved edited content to {new_file_path}")

# Functions to handle checkpoint file
def load_checkpoint():
    global last_processed_file, last_processed_prompt
    try:
        with open("checkpoint.txt", "r") as checkpoint_file:
            last_processed_data = checkpoint_file.read().strip().split(',')
            if len(last_processed_data) == 2:
                last_processed_file, last_processed_prompt = last_processed_data[0], int(last_processed_data[1])
            else:
                last_processed_file, last_processed_prompt = last_processed_data[0], 0
    except FileNotFoundError:
        pass

def update_checkpoint(filename, prompt_number):
    global last_processed_file, last_processed_prompt
    last_processed_file = filename
    last_processed_prompt = prompt_number
    with open("checkpoint.txt", "w") as checkpoint_file:
        checkpoint_file.write(f"{filename},{prompt_number}")

if __name__ == "__main__":
    main()
