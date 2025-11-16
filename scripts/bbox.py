import os
import json
import cv2
import numpy as np
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(repo_id="AdamCodd/YOLOv11n-face-detection", filename="model.pt")
model = YOLO(model_path)  

class ManualBBoxAnnotator:
    def __init__(self):
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.bbox = []
        self.window_name = "Manual BBox Annotation - Draw rectangle and press SPACE to confirm, ESC to skip"
    
    def draw_rectangle(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                img_copy = self.img.copy()
                cv2.rectangle(img_copy, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
                cv2.imshow(self.window_name, img_copy)
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            x1, y1 = min(self.ix, x), min(self.iy, y)
            x2, y2 = max(self.ix, x), max(self.iy, y)
            self.bbox = [x1, y1, x2, y2]
            
            # Show final bbox
            img_copy = self.img.copy()
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_copy, "Press SPACE to confirm, ESC to skip", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(self.window_name, img_copy)
    
    def annotate(self, img_path):
        self.img = cv2.imread(img_path)
        if self.img is None:
            print(f"Could not load image: {img_path}")
            return None
        
        self.bbox = []
        img_display = self.img.copy()
        cv2.imshow(self.window_name, img_display)
        cv2.setMouseCallback(self.window_name, self.draw_rectangle)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # SPACE to confirm
                if self.bbox:
                    cv2.destroyWindow(self.window_name)
                    return self.bbox
                else:
                    print("Please draw a bounding box first")
            elif key == 27:  # ESC to skip
                cv2.destroyWindow(self.window_name)
                return None
        
        cv2.destroyWindow(self.window_name)
        return self.bbox

def parse_filename(filename):
    """Парсинг имени файла для извлечения ID и имени правителя"""
    if '_obv_' in filename:
        id_part, emperor = filename.replace('_obv_', '_').replace('.jpg', '').split('_')[:2]
        return id_part, emperor, 'obv'
    else:
        parts = filename.replace('.jpg', '').split('_')
        if len(parts) >= 2:
            return parts[0], parts[1], 'rev'
        else:
            return parts[0], "unknown", 'rev'

def process_images(image_dir):
    obv_results = {}
    rev_results = {}
    manual_annotator = ManualBBoxAnnotator()
    
    for filename in os.listdir(image_dir):
        if not filename.endswith('.jpg'):
            continue
            
        print(f"Processing: {filename}")
        id_part, emperor, img_type = parse_filename(filename)
        
        img_path = os.path.join(image_dir, filename)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Could not load image: {filename}")
            continue
        
        results = model(img)
        
        bbox_data = None
        confidence = 0.0
        
        if len(results[0].boxes) > 0:
            best_box = None
            max_conf = 0
            for box in results[0].boxes:
                if box.conf > max_conf:
                    max_conf = box.conf
                    best_box = box.xyxy[0].tolist()
            
            bbox_data = best_box
            confidence = max_conf.item()
            print(f"Auto-detected bbox with confidence: {confidence:.3f}")
        else:
            print("No detection found. Manual annotation required.")
            bbox_data = manual_annotator.annotate(img_path)
            confidence = 1.0 
            if bbox_data:
                print(f"Manual bbox: {bbox_data}")
            else:
                print("Image skipped by user")
                continue
     
        result_data = {
            'filename': filename,
            'bbox': bbox_data,
            'confidence': confidence
        }
        
        if img_type == 'obv':
            if emperor not in obv_results:
                obv_results[emperor] = []
            obv_results[emperor].append(result_data)
        else:
            if emperor not in rev_results:
                rev_results[emperor] = []
            rev_results[emperor].append(result_data)
    

    for emperor, data in obv_results.items():
        output_filename = f'bbox_obv_{emperor}.json'
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(data)} obv results to {output_filename}")
            
    for emperor, data in rev_results.items():
        output_filename = f'bbox_rev_{emperor}.json'
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(data)} rev results to {output_filename}")

def main():
    image_dir = './images_300'
    
    if not os.path.exists(image_dir):
        print("Directory does not exist!")
        return
    
    print("Starting image processing...")
    print("Manual annotation instructions:")
    print("- Draw rectangle with mouse")
    print("- Press SPACE to confirm")
    print("- Press ESC to skip image")
    print("- Close window to cancel current annotation")
    
    process_images(image_dir)
    print("Processing completed!")

if __name__ == "__main__":
    main()