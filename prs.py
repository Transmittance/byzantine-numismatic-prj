from genericpath import exists
from selenium import webdriver 
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import requests, os, time, re, csv, datetime, random
from urllib.parse import urljoin
import xml.etree.ElementTree as ET
import json
from urllib.parse import unquote
import time
import csv as cs
OUTPUT_DIR_IMAGES = "images"
OUTPUT_DIR_XMLS = "xmls"
OUTPUT_DIR_JSONS = 'jsons'
csv = "main.csv"
TIMEOUT = 30
MAX_RETRIES = 3
os.makedirs(OUTPUT_DIR_IMAGES, exist_ok=True)
os.makedirs(OUTPUT_DIR_XMLS, exist_ok=True)
os.makedirs(OUTPUT_DIR_JSONS, exist_ok=True)
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

def get_safe_filename(name):
    invalid_chars = r'[<>:"/\\|?*\n\r\t]'
    safe_name = re.sub(invalid_chars, '_', name)

    safe_name = safe_name.rstrip('.')

    safe_name = safe_name[:100]

    if not safe_name.strip():
        safe_name = f"document_{int(time.time())}"

    return safe_name

def parse_nuds_xml_to_json(xml_string):
    namespaces = {
        'nuds': 'http://nomisma.org/nuds',
        'mets': 'http://www.loc.gov/METS/',
        'xlink': 'http://www.w3.org/1999/xlink'
    }
    
    root = ET.fromstring(xml_string)
    try:
        record_id_elem = root.find('.//nuds:recordId', namespaces)
        record_id = record_id_elem.text if record_id_elem is not None else "Unknown"
        
        title_elem = root.find('.//nuds:title', namespaces)
        title = title_elem.text if title_elem is not None else "Unknown"

        coin_page_url = f"https://numismatics.org/collection/{record_id}" if record_id != "Unknown" else "Unknown"
        

        date_from_elem = root.find('.//nuds:fromDate', namespaces)
        date_from = date_from_elem.get('standardDate') if date_from_elem is not None else None
        date_to_elem = root.find('.//nuds:toDate', namespaces)
        date_to = date_to_elem.get('standardDate') if date_to_elem is not None else None

        material_elem = root.find('.//nuds:material', namespaces)
        material = material_elem.get(f"{{{namespaces['xlink']}}}href") if material_elem is not None else None
        
        denomination_elem = root.find('.//nuds:denomination', namespaces)
        denomination = denomination_elem.get(f"{{{namespaces['xlink']}}}href") if denomination_elem is not None else None
        
        obverse_desc_elem = root.find('.//nuds:obverse/nuds:type/nuds:description', namespaces)
        obverse_desc = obverse_desc_elem.text if obverse_desc_elem is not None else None
        
        reverse_desc_elem = root.find('.//nuds:reverse/nuds:type/nuds:description', namespaces)
        reverse_desc = reverse_desc_elem.text if reverse_desc_elem is not None else None
        
        obverse_portrait_elem = root.find('.//nuds:obverse/nuds:persname', namespaces)
        obverse_portrait = obverse_portrait_elem.text if obverse_portrait_elem is not None else None
        
        refDesc_elem = root.find('.//nuds:refDesc/nuds:reference', namespaces)
        refDesc = refDesc_elem.text if refDesc_elem is not None else None
        
        symbol_elem = root.find('.//nuds:reverse/nuds:symbol', namespaces)
        symbol = symbol_elem.text if symbol_elem is not None else None
        
        legend_elem = root.find('.//nuds:obverse/nuds:legend', namespaces)
        legend = legend_elem.text if legend_elem is not None else None
        

        weight_elem = root.find('.//nuds:weight', namespaces)
        weight = weight_elem.text if weight_elem is not None else "Unknown"
        
        diameter_elem = root.find('.//nuds:diameter', namespaces)
        diameter = diameter_elem.text if diameter_elem is not None else "Unknown"
        
        images = {
            'obverse': [],
            'reverse': []
        }
        
        for file_grp in root.findall('.//mets:fileGrp', namespaces):
            usage = file_grp.get('USE')
            if usage in ['obverse', 'reverse']:
                for file_elem in file_grp.findall('mets:file', namespaces):
                    file_use = file_elem.get('USE')
                    flocat = file_elem.find('mets:FLocat', namespaces)
                    if flocat is not None:
                        url = unquote(flocat.get(f"{{{namespaces['xlink']}}}href"))
                        images[usage].append({
                            'type': file_use,
                            'url': url
                        })
    except Exception as e:
        print(f"Ошибка при парсинге XML: {str(e)}")    
    result = {
        'metadata': {
            'record_id': record_id,
            'coin_page_url': coin_page_url,
            'title': title,
            'date_range': {
                'from': date_from,
                'to': date_to
            },
            'obverse_portrait': obverse_portrait,
            'material': material,
            'denomination': denomination,
            'cat': refDesc,
            'Obverse_leg': legend,
            'rev_symbol': symbol,
            'measurements': {
                'weight': f"{weight} g",
                'diameter': f"{diameter} mm"
            },
            'descriptions': {
                'obverse': obverse_desc,
                'reverse': reverse_desc
            }
        },
        'images': images
    }
    
    return json.dumps(result, indent=2, ensure_ascii=False)
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
            time.sleep(1)
            if response.status_code != 200:
                print(f"Ошибка HTTP {response.status_code} при попытке {attempt + 1}")
                continue
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            if os.path.getsize(filepath) == 0:
                print(f"Скачан пустой файл при попытке {attempt + 1}")
                time.sleep(1)
                os.remove(filepath)
                continue
            return True
        except Exception as e:
            print(f"Ошибка при скачивании файла {url} (попытка {attempt +1}): {str(e)}")
    return False
def process_page(csv_writer, index, row_reading):

    items = wait.until(EC.presence_of_all_elements_located(
        (By.XPATH, "//a[@rel='nofollow']")
    ))
    time.sleep(1)
    links = [item.get_attribute("href") for item in items]

    for link in links:
        if link.endswith('.xml'):
            filename = f"{index}_{row_reading}.xml"
            filepath = os.path.join(OUTPUT_DIR_XMLS, get_safe_filename(filename))
            success = file_download(link, filepath)
            if success:
                with open(filepath, 'r', encoding='utf-8') as fl:
                    xml_string = fl.read()
                json_string = parse_nuds_xml_to_json(xml_string)
                json_data = json.loads(json_string)

                url_image_obverse = [img for img in json_data['images']['obverse'] if img['type'] == 'archive']
                url_image_reverse = [img for img in json_data['images']['reverse'] if img['type'] == 'archive']
                coin_page_url = json_data['metadata']['coin_page_url']
                portrait = json_data['metadata']['obverse_portrait']

                filename_1 = f"{index}_obv_{row_reading}.jpg"
                filepath_1 = os.path.join(OUTPUT_DIR_IMAGES, get_safe_filename(filename_1))
                filename_2 = f"{index}_rev_{row_reading}.jpg"
                filepath_2 = os.path.join(OUTPUT_DIR_IMAGES, get_safe_filename(filename_2))
                success_obv = file_download(url_image_obverse[0]['url'], filepath_1)
                success_rev = file_download(url_image_reverse[0]['url'], filepath_2)

                if success_obv and success_rev:
                    filename_json = f"{index}_{row_reading}.json"
                    filepath_json = os.path.join(OUTPUT_DIR_JSONS, get_safe_filename(filename_json))
                    with open(filepath_json, 'w', encoding='utf-8') as f:
                        f.write(json_string)
                    return True, coin_page_url, portrait
                else:
                    print(f"Не удалось скачать файлы")
    return False


with open(csv, 'w', newline='', encoding='utf-8') as csvfile:
    csv_wr = cs.writer(csvfile)
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
    files = ['csv/Const I.csv', 'csv/Theodosius I.csv', 'csv/Theodosius II.csv']
    index = 0
    for file in files:
        with open(file, 'r', encoding='utf-8') as fi:
            csv_r = cs.DictReader(fi)
            for row in csv_r:
                try:
                    driver.get(row['URI'])
                    success, coin_url, portrait = process_page(csv_wr, index, row["Портрет"])
                    if success:
                        image_av = f"{index}_obv_{row["Портрет"]}.jpg"
                        image_rev = f"{index}_rev_{row["Портрет"]}.jpg"
                        meta_file = f"{index}_{row["Портрет"]}.json"
                        csv_wr.writerow([
                            index,
                            row["Портрет"],
                            os.path.join(OUTPUT_DIR_IMAGES, image_av),
                            os.path.join(OUTPUT_DIR_IMAGES, image_rev),
                            os.path.join(OUTPUT_DIR_JSONS, meta_file),
                            '',
                            '',
                            coin_url
                        ])
                        index+=1
                except Exception as e: 
                    print(f'Ошибка {e}')
                    continue
driver.quit()