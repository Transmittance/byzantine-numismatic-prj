import pandas as pd
import json

main_df = pd.read_csv('main.csv')

with open('./bboxes/merged_bbox_obv.json', 'r') as f:
    bbox_data = json.load(f)

bbox_map = {}
for entry in bbox_data:
    filename = entry['filename']
    file_id = filename.split('_')[0]
    bbox_map[file_id] = entry['bbox']

def get_bbox_for_id(row):
    bbox = bbox_map.get(str(row['ID']), None)
    if bbox is not None:
        return json.dumps(bbox)
    return None

main_df['bbox_av_face'] = main_df.apply(get_bbox_for_id, axis=1)
main_df.to_csv('main_updated.csv', index=False)