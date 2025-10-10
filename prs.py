from selenium import webdriver 
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import requests, os, time, re, csv, datetime, random
from urllib.parse import urljoin
OUTPUT_DIR = "images"
csv = "main.csv"
TIMEOUT = 30
MAX_RETRIES = 3
os.makedirs(OUTPUT_DIR, exist_ok=True)

options = webdriver.ChromeOptions()
options.add_experimental_option("prefs", {
    "download.default_directory": os.path.abspath(OUTPUT_DIR),
    "download.prompt_for_download": False,
    "download.directory_upgrade": True,
    "plugins.always_open_pdf_externally": True,
    "safebrowsing.enabled": True
})

driver = webdriver.Chrome(options = options)
driver.maximize_window()
wait =WebDriverWait(driver, TIMEOUT)

def process_page(csv_writer):
    items = wait.until(EC.presence_of_all_elements_located(
        (By.CLASS_NAME, "")
    ))

with open(csv, 'w', newline='', encoding='utf-8') as csvfile:
    csv_wr = csv.writer(csvfile)
    csv_wr.writerow([
        'ID',
        'Imperor', 
        'Image_path_av',
        'Image_path_rev',
        'meta',
        'bbox_av_face', 
        'bbox_rev_img',
        'URL'
    ])
    files = ['Const I.csv', 'Theodosius I.csv', 'Theodosius II.csv']
    for file in files:
        with open(file, 'r'):
            csv_r = csv.reader(file)
            for row in csv_r:
                try:
                    driver.get(row['URI'])
                    process_page(csv_wr)
                except: break