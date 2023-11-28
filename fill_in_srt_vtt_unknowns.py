# Description: This script reads the TXT, VTT, and SRT files for each country and replaces the <unk> tags in the VTT and SRT files with the words from the TXT file.
# The file structure is as follows:
#
# txt: _trimmed_trimmed.txt
# vtt: _78_trimmed_combined.vtt
# srt: _78_trimmed_combined.srt


import os
import re

def replace_unk_tags(txt_content, vtt_content, srt_content):
    # Find <unk> tags in VTT and SRT files and replace them with words from the TXT file
    vtt_content_modified = re.sub(r'<unk>', lambda match: find_word_context(match.group(0), txt_content), vtt_content)
    srt_content_modified = re.sub(r'<unk>', lambda match: find_word_context(match.group(0), txt_content), srt_content)

    return vtt_content_modified, srt_content_modified

def find_word_context(unk_tag, txt_content):
    # Find the index of <unk> in the list of words
    words = txt_content.split()
    
    try:
        unk_index = words.index('<unk>')
        print(f'Found <unk> at index {unk_index}')

        # Replace <unk> with the word before it and the word after it
        if unk_index > 0 and unk_index < len(words) - 1:
            return words[unk_index - 1] + ' ' + words[unk_index + 1]
        elif unk_index == 0 and len(words) > 1:
            return words[unk_index + 1]
        elif unk_index == len(words) - 1 and len(words) > 1:
            return words[unk_index - 1]
        else:
            # If <unk> is the only word or not found, return an empty string
            return ''
    except ValueError:
        # Handle the case when '<unk>' is not in the list
        return ''


def process_files(txt_folder, vtt_folder, srt_folder, output_folder):
    # List all files in the specified folders
    txt_files = [f for f in os.listdir(txt_folder) if f.endswith('_trimmed_trimmed.txt')]
    vtt_files = [f for f in os.listdir(vtt_folder) if f.endswith('_78_trimmed_combined.vtt')]
    srt_files = [f for f in os.listdir(srt_folder) if f.endswith('_78_trimmed_combined.srt')]

    # Process each file
    for txt_filename in txt_files:
        print(f'Processing {txt_filename}')
        # Extract the country name from the TXT file name, its the first word before the first underscore
        match = txt_filename.split('_')[0]
        print(f'Country: {match}')
        if match:
            country = match
            vtt_filename = f'{country}_78_trimmed_combined.vtt'
            srt_filename = f'{country}_78_trimmed_combined.srt'

            if vtt_filename in vtt_files and srt_filename in srt_files:
                # Read content from the TXT, VTT, and SRT files
                with open(os.path.join(txt_folder, txt_filename), 'r') as txt_file:
                    txt_content = txt_file.read()

                with open(os.path.join(vtt_folder, vtt_filename), 'r') as vtt_file:
                    vtt_content = vtt_file.read()

                with open(os.path.join(srt_folder, srt_filename), 'r') as srt_file:
                    srt_content = srt_file.read()

                # Replace <unk> tags in VTT and SRT files
                vtt_content_modified, srt_content_modified = replace_unk_tags(txt_content, vtt_content, srt_content)

                # Save modified content to new files in the output folder
                new_vtt_filename = os.path.join(output_folder, f'modified_{vtt_filename}')
                new_srt_filename = os.path.join(output_folder, f'modified_{srt_filename}')

                with open(new_vtt_filename, 'w') as new_vtt_file:
                    new_vtt_file.write(vtt_content_modified)
                    print(f'Modified VTT content saved to {new_vtt_filename}')

                with open(new_srt_filename, 'w') as new_srt_file:
                    new_srt_file.write(srt_content_modified)
                    print(f'Modified SRT content saved to {new_srt_filename}')

            else:
                if vtt_filename not in vtt_files:
                    print(f"VTT file not found for {txt_filename}: {vtt_filename}")
                if srt_filename not in srt_files:
                    print(f"SRT file not found for {txt_filename}: {srt_filename}")
        else:
            print(f"No match found for country pattern in {txt_filename}")

def main():
    # Replace these folder paths with the actual paths
    txt_folder = '/media/rusty/Data2/UNGA/UNGA_78/txt'
    vtt_folder = '/media/rusty/Data2/UNGA/UNGA_78/vtt'
    srt_folder = '/media/rusty/Data2/UNGA/UNGA_78/srt'
    output_folder = '/media/rusty/Data2/UNGA/UNGA_78/output'

    process_files(txt_folder, vtt_folder, srt_folder, output_folder)

if __name__ == '__main__':
    main()
