import os, re, sys
import shutil
import pandas as pd
from pathlib import Path
from PIL import Image

def parse_bbox_any(cell):
    s = str(cell)
    nums = re.findall(r"-?\d+(?:\.\d+)?", s)
    if len(nums) >= 4:
        x1, y1, x2, y2 = map(float, nums[:4])
        return x1, y1, x2, y2
    return None

def clamp_xyxy(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(round(x1)), w-1))
    x2 = max(0, min(int(round(x2)), w-1))
    y1 = max(0, min(int(round(y1)), h-1))
    y2 = max(0, min(int(round(y2)), h-1))
    if x2 <= x1: x2 = min(w-1, x1+1)
    if y2 <= y1: y2 = min(h-1, y1+1)
    return x1, y1, x2, y2

df = pd.read_csv("./main_updated.csv")

# Выходная папка
os.makedirs("./images_by_classes", exist_ok=True)

for _, row in df.iterrows():
    rel_path = str(row["Image_path_av"]).replace("\\", "/").replace("||", "__")

    img_path = Path("./parsed_data") / rel_path

    if not img_path.exists():
        print(f"Файл не найден: {img_path}")
        continue

    imperor = str(row["Imperor"]).strip()

    class_dir = Path("./images_by_classes") / imperor
    class_dir.mkdir(parents=True, exist_ok=True)

    dst_path = class_dir / img_path.name

    # Парсим ббоксы
    bbox = parse_bbox_any(row.get("bbox_av_face")) if "bbox_av_face" in df.columns else None

    # Кроп по ббоксу
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        w, h = im.size
        x1, y1, x2, y2 = clamp_xyxy(*bbox, w=w, h=h)
        im.crop((x1, y1, x2, y2)).save(dst_path)
    continue

# id_rev and id_obv -> parsed_data -> images
# bbox: bbox_obv_name.json -> bbox_jsons_by_emperor 
