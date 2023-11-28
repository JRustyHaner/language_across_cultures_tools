#Alistair Windsor's code for scraping the UN General Debate videos

import csv
import os
import re
import requests
#import shutil
#import time
from bs4 import BeautifulSoup
from bs4 import UnicodeDammit
import json

# Selenium block
from selenium import webdriver 
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import selenium.common.exceptions as sel_exceptions
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


#from unidecode import unidecode

base_dir = '/home/rusty/Desktop/workspace/language_across_cutures/UNGA_78/'
out_filename = 'UNGeneralDebateSearchResults78.csv'
out_dir = '/home/rusty/Desktop/workspace/language_across_cutures/UNGA_78/'
out_filename2 ='UNGeneralDebateSearchResults78_2.csv'
search = [
        {'session': 78, 'url' :'https://www.unmultimedia.org/avlibrary/search/search.jsp?fromManualDate=2023-09-19&toManualDate=2023-09-26&category=GENERAL+ASSEMBLY+MEETING&q=&sort=cdate_desc&lang=&start='}
        ]
def find_multimedia_pages(search):
    os.chdir(base_dir)
    with open(out_filename,'w') as out_file:
        out_csv = csv.writer(out_file)
        out_csv.writerow(['session','country','Search Page URL','Video Title','Video Description','Link'])
        for entry in search:
            start = 0
            while True:
                search_url = entry['url'] + str(start)
                r= requests.get(search_url)
                soup = BeautifulSoup(UnicodeDammit.detwingle(r.content))
                for tag in soup.find_all("a",class_="search_desc_link"):
                    link=tag['href']
                    title = tag.find("div",class_="search_pg_video_title").get_text()
                    country = re.search(r'(.*)[-–] .*',title)
                    if country:
                        country = country.group(1)
                    else:
                        country = ""
                    if title.startswith('Lao People\'s Democratic Republic'):
                        country = "Lao People's Democratic Republic"
                    desc = tag.find("div", class_="search_pg_video_desc moresearchpage").get_text()
                    if re.search(r'General Debate', title):
                        out_csv.writerow([entry['session'],country,search_url , title, desc, link])
                next_pageQ = soup.find("a", title="Go to next page")
                if not next_pageQ:
                    print("End of search results")
                    break
                else:
                    start += 10

def kaltura_direct_download_link(PartnerId, EntryId, VideoFlavorId = 0):
    serviceUrl = "https://cdnapisec.kaltura.com"
    playManifestAPI = f"{serviceUrl}/p/{PartnerId}/sp/0/playManifest/entryId/{EntryId}/format/url/flavorParamId/{VideoFlavorId}/video.mp4"
    return playManifestAPI

def DetermineIDs(link):
    # We use a remote driver, at http://iisml-precision-7920-tower.uom.memphis.edu:4444
    # This is a Selenium Grid server running on a Ubuntu 20.04 Machine
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--headless')
    driver = webdriver.Remote(command_executor='http://iisml-precision-7920-tower.uom.memphis.edu:4444',
                              options=chrome_options)
    print("Getting link: ", link)
    driver.get(link)
    with open('page_source.html','w') as out_file:
        print("Writing page source to file")
        out_file.write(driver.page_source)
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "iframe")))
    #write the page source to a file
    r = requests.get(link)
    parsed_page = BeautifulSoup(r.text, features="html.parser")
    iframe = parsed_page.find('iframe')
    kaltura_link = iframe['src']
    PartnerId = re.search(r'partner_id/(\d*)[\?/]',kaltura_link).group(1)
    EntryId = re.search(r'entry_id=([0-9a-z_]+)[\?/]?',kaltura_link).group(1)
    VideoFlavorId = re.search(r'flavorIds/([0-9a-z_,]+)/',kaltura_link)
    print(f'PartnerId: {PartnerId}, EntryId: {EntryId}, VideoFlavorId: {VideoFlavorId}')
    if VideoFlavorId:
        VideoFlavorId = VideoFlavorId.group(1)
    else:
        print(link)
        VideoFlavorId=0
    print("Finished getting IDs")
    driver.quit()
    return PartnerId, EntryId, VideoFlavorId

def DetermineVideoLinks():
    os.chdir(base_dir)
    with  open(out_filename,'r') as in_file, \
          open(out_filename2,'w') as out_file:
        in_csv = csv.DictReader(in_file)
        out_fields = in_csv.fieldnames+['PartnerId','EntryId','Video Download Link']
        out_csv = csv.DictWriter(out_file,fieldnames=out_fields)
        out_csv.writeheader()
        previous_session=None
        for row in in_csv:
            print(f'Processing {row["Video Title"]}')
            #check if that file exists. If so skip it
            if os.path.isfile(f'{row["session"]}_{row["country"]}.mp4'):
                print(f'Video for {row["country"]} in session {row["session"]} already exists. Skipping')
                continue
            # tell us how far along we are
            PartnerId, EntryId, VideoFlavorId= DetermineIDs(row['Link'])
            videoURL = kaltura_direct_download_link(PartnerId, EntryId, VideoFlavorId)
            row['PartnerId']=PartnerId
            row['EntryId'] = EntryId
            row['Video Download Link'] = videoURL
            out_csv.writerow(row)
            # download the video using requests
            #check if the video already exists
            print(f'Downloading video for {row["country"]} in session {row["session"]}')
            r = requests.get(videoURL, stream=True)
            with open(f'{row["session"]}_{row["country"]}.mp4', 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            print(f'Finished downloading video for {row["country"]} in session {row["session"]}')

            

def check_dir(dir):
    """
    Switch to the given directory if it exists, if not create it and then switch.
    """
    if not os.path.isdir(dir):
        os.mkdir(dir)
    os.chdir(dir)

def get_valid_filename(name):
    """
    Return the given string converted to a string that can be used for a clean
    filename. Remove leading and trailing spaces; convert other spaces to
    underscores; and remove anything that is not an alphanumeric, dash,
    underscore, or dot.
    >>> get_valid_filename("john's portrait in 2004.jpg")
    'johns_portrait_in_2004.jpg'
    """
    s = str(name).strip().replace(' ', '_')
    s = re.sub(r'(?u)[^-\w.]', '', s)
    if s in {'', '.', '..'}:
        raise Exception("Could not derive file name from '%s'" % name)
    return s

def DownloadVideos():
    os.chdir(base_dir)
    with open(out_filename2,'r') as in_file:
        in_csv = csv.DictReader(in_file)
        for row in in_csv:
            session = row['session']
            session_dir = os.path.join(out_dir,session)
            check_dir(session_dir)
            country = row['country']
            if country != "":
                country_filename = get_valid_filename(country)
            else:
                country_filename = get_valid_filename(row['Video Title'])
            if not os.path.isfile(f'{session}_{country_filename}.mp4'):
                print(f'Downloading video for {country_filename} in session {session}')
                os.system(f'curl {row["Video Download Link"]} -o {session}_{country_filename}.mp4')

def clear_all_sessions():
    url="http://iisml-precision-7920-tower.uom.memphis.edu:4444"
    # delete all sessions
    r = requests.get("{}/status".format(url))
    data = json.loads(r.text)
    for node in data['value']['nodes']:
        for slot in node['slots']:
            if slot['session']:
                id = slot['session']['sessionId']
                r = requests.delete("{}/session/{}".format(url, id))

clear_all_sessions()
clear_all_sessions()
DetermineVideoLinks()