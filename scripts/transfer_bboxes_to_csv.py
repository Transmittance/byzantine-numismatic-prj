import pandas as pd
import os
import json
import glob
import csv

jsons_folder = './bboxes/bbox_jsons_by_emperor/'
merged_json_path = './bboxes/merged_bbox_obv.json'

def parse_jsons(folder, merged_path):
    json_data  = []
    json_files = [file for file in glob.glob(os.path.join(folder, "*.json")) if '_obv_' in os.path.basename(file)]
    
    for json_path in json_files:
        with open(json_path, 'r', encoding='utf-8') as fj:
            data = json.load(fj)
            if isinstance(data, list):
                json_data.extend(data)
            else:
                print(f'{json_path} пустой')
    with open(merged_path, 'w', encoding='utf-8') as fm:
        json.dump(json_data, fm, indent=2)
    
parse_jsons(jsons_folder, merged_json_path)


with open(merged_json_path, 'r') as f:
    merged_json = json.load(f)

bbox_map, imperors_map, image_path_av_map = {}, {}, {}
file_ids = []

imperors_names = ['Severus Alexander', 
                  'Caligula', 
                  'Commodus',
                  'Domitian', 
                  'Constantine I', 
                  'Constantius Chlorus', 
                  'Diocletian',
                  'Hadrian', 
                  'Marcus Aurelius', 
                  'Octavian',
                  'Septimius Severus',
                  'Theodosius I',
                  'Theodosius II',
                  'Tiberius',
                  'Vespasian']

# filename = 762_obv_Theodosius II||Arcadius||Honorius.jpg 
# if 'Theodosius II' in filename:
#   imperor = 'Theodosius II'

for entry in merged_json:
    filename = entry['filename']
    file_id = filename.split('_')[0]

    matches_count = 0
    imperor = None
    for name in imperors_names:
        if name in filename:
            imperor = name
            matches_count = matches_count + 1 if imperor not in ('Theodosius I', 'Theodosius II') else matches_count
            if matches_count > 1:
                break

    if matches_count > 1:
        continue
    
    bbox_map[file_id] = entry['bbox']
    imperors_map[file_id] = imperor
    image_path_av_map[file_id] = 'images/' + filename
    
    file_ids.append(file_id)

fields = ["ID","Imperor","Image_path_av","bbox_av_face"]
data = []
for file_id in file_ids:
    data.append({
        "ID": file_id,
        "Imperor": imperors_map.get(file_id, ""),
        "Image_path_av": image_path_av_map.get(file_id, ""),
        "bbox_av_face": ""
    })

data.sort(key = lambda x: int(x["ID"]))

with open('full_main.csv', 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=fields)
    
    writer.writeheader()
    writer.writerows(data)

def get_bbox_for_id(row):
    bbox = bbox_map.get(str(row['ID']), None)
    if bbox is not None:
        return json.dumps(bbox)
    return None

main_df = pd.read_csv('full_main.csv')
main_df['bbox_av_face'] = main_df.apply(get_bbox_for_id, axis=1)
main_df.to_csv('main_updated.csv', index=False)