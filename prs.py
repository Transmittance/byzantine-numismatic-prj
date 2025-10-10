from genericpath import exists
from selenium import webdriver 
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import requests, os, time, re, csv, datetime, random
from urllib.parse import urljoin
OUTPUT_DIR_IMAGES = "images"
OUTPUT_DIR_XMLS = "images"
csv = "main.csv"
TIMEOUT = 30
MAX_RETRIES = 3
os.makedirs(OUTPUT_DIR_IMAGES, exist_ok=True)
os.makedirs(OUTPUT_DIR_XMLS, exists_ok=True)
options = webdriver.ChromeOptions()
options.add_experimental_option("prefs", {
    "download.default_directory": os.path.abspath(OUTPUT_DIR_IMAGES),
    "download.prompt_for_download": False,
    "download.directory_upgrade": True,
    "plugins.always_open_pdf_externally": True,
    "safebrowsing.enabled": True
})

driver = webdriver.Chrome(options = options)
driver.maximize_window()
wait =WebDriverWait(driver, TIMEOUT)
def file_download(url, filepath, max_retries=MAX_RETRIES):
    for attempt in range(max_retries):
        try:
            session = requests.Session()
            selenium_cookies = driver.get_cookies()
            for cookie in selenium_cookies:
                session.cookies.set(cookie['name'], cookie['value'])
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            response = session.get(url, headers=headers, stream=True, timeout=30)
            if response.status_code != 200:
                print(f"Ошибка HTTP {response.status_code} при попытке {attempt + 1}")
                continue
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            if os.path.getsize(filepath) == 0:
                print(f"Скачан пустой файл при попытке {attempt + 1}")
                os.remove(filepath)
                continue
            return True
        except Exception as e:
            print(f"Ошибка при скачивании файла {url} (попытка {attempt +1}): {str(e)}")
    return False
def process_page(csv_writer, index, row_reading):
    items = wait.until(EC.presence_of_all_elements_located(
        (By.XPATH, "//a[@rel=’nofollow’]")
    ))
    links = [item.get_attribute("href") for item in items]
    for link in links:
        if link[::4] == '.xml':
            filename = f"{index}_{row_reading}.xml"
            filepath = os.path.join(OUTPUT_DIR_XMLS, filename)
            success = file_download(link, filepath)
            if success:
                #добавить распарсинг xml файла, который будет скачан по монетке (файл находится на сайте на страничке монетки сверху в шапке: EXPORT: NUDS/XML)

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
            for index, row in enumerate(csv_r):
                try:
                    driver.get(row['URI'])
                    process_page(csv_wr, index, row['Портрет'])
                except: break