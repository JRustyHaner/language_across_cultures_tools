# Language Across Cultures Repository


This repository contains a collection of scripts developed at the Institute of Intelligent Systems at the University of Memphis for the Language Across Cultures project. These scripts cover a range of functionalities related to language processing, data extraction, and video manipulation.


## Scripts Overview


1. **automate_queries_to_chatgpt_and_save_in_csv.py**

   - **Purpose:** Generate responses using the ChatGPT API and save them in a CSV file.

   - **Dependencies:**

       - requests

       - json

       - csv

       - time

       - openai


2. **awindsor_scrapeVideo.py**

   - **Purpose:** Alistair Windsor's code designed for scraping UN General Debate videos.

   - **Dependencies:**

       - csv

       - os

       - re

       - requests

       - bs4 (BeautifulSoup)

       - UnicodeDammit

       - json


3. **combine_mkv_audio_and_video.py**

   - **Purpose:** Replace the audio in a video file with the audio from another video file.

   - **Dependencies:**

       - os

       - moviepy.editor (specifically VideoFileClip and AudioFileClip)

4. **combine_cohmetrix_csvs.py**

   - **Purpose:** Combines output from multiple cohmetrix csvs to one.

   - **Dependencies:**

       - os

       - moviepy.editor (specifically VideoFileClip and AudioFileClip)


5. **convert_mkv_to_mp4.py**

   - **Functionality:** Converts MKV files to MP4 files using the moviepy library.

   - **Dependencies:**

       - os

       - moviepy.editor (specifically VideoFileClip)


6. **convert_mp4_to_wmv_and_audio.py**

   - **Purpose:** Convert .mp4 files in a folder to .wmv and .mp3 files.

   - **Dependencies:**

       - os

       - subprocess

       - moviepy.editor


7. **convert_wmv_to_mp3_audio.py**

   - **Purpose:** Convert .wmv files in a folder to .mp3 files.

   - **Dependencies:**

       - os

       - subprocess

       - moviepy.editor


8. **dedirectoryify.py**

   - **Purpose:** Move files from subdirectories to the current directory.

   - **Dependencies:**

       - os

       - shutil

       - sys


9. **directify_by_type.py**

   - **Purpose:** Organize files by moving them to folders based on their file extensions.

   - **Dependencies:**

       - os

       - shutil


10. **extract_csv_column_to_txt.py**

   - **Functionality:** Extract a specified column from a CSV file and write it to a text file.

   - **Dependencies:**

       - os

       - re

       - sys

       - datetime

       - pandas


11. **fill_in_srt_vtt_unknowns.py**

   - **Purpose:** Replace <unk> tags in VTT and SRT files with words from the corresponding TXT file.

   - **Dependencies:**

       - os

       - re


12. **generate_json_and_srt_captions.py**

   - **Functionality:** Generate JSON and SRT files for all MP3 files in the MP3 folder using the Gentle server.

   - **Dependencies:**

       - os

       - subprocess

       - json

       - gentle server


13. **group_srt_subtitles_into_short_sentences.py**

   - **Purpose:** Combine subtitle entries based on specified conditions.

   - **Conditions:**

       - Combine if the length is less than MAX_ENTRY_LENGTH.

       - Combine if the start time is less than 5 seconds after the end time of the previous entry.

   - **Dependencies:**

       - os

       - re

       - datetime


14. **read_pdf_with_OCR.py**

   - **Description:** Extract text from PDF files using PyMuPDF and Tesseract OCR.

   - **Dependencies:**

       - os

       - fitz (PyMuPDF)

       - re

       - unicodedata

       - PIL (Image)

       - pytesseract

15. **remove_background_from_video_pytorch.py**

   - **Description:** Removes the background of a video and replaces it with no backgound
   uses pytorch. 

   - **Dependencies:**

       - please see the script

15. **remove_background_from_video.py**

   - **Description:** Removes the background of a video and replaces it with rgb green

   - **Dependencies:**

       - please see the script


16. **scrape_pdf.py**

   - **Description:** Download PDF files from the UN website and save them to a directory named after the country code.

   - **Dependencies:**

       - requests

       - os


17. **skinColorClassifier.py**

   - **Description:** Classify skin color on the Von Luschan scale using average pixel color of an image.

   - **Dependencies:**

       - os

       - re

       - numpy

       - scipy.spatial.distance

       - PIL (Image)

       - cv2

       - sys

       - scipy.ndimage


18. **superimpose_captions_on_wmv.py**

   - **Description:** Superimpose captions on WMV video files using SRT files.

   - **Dependencies:**

       - moviepy.editor (VideoFileClip, TextClip, CompositeVideoClip)

       - pysrt

       - os

19. **get_highest_imotions_values_from_wmv**

    - **Description:** Gets an image of the frame with the highest emotion values given a video and imotions csv data

   - **Dependencies:**

       - pandas, numpy, cv2
