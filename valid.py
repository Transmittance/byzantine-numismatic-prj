import json
import os
from tkinter import image_names
import cv2 

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
def draw_bbox(image_path, list):
    window_name = 'bbox_validation'
    img = cv2.imread(image_path)
    img_copy = img.copy()
    cv2.rectangle(img_copy, (int(list[0]), int(list[1])), (int(list[2]), int(list[3])), (0, 255, 0), 2)
    cv2.putText(img_copy, "Press SPACE to annotate, ESC to skip", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow(window_name, img_copy)
    key = cv2.waitKey(0)
    bbox_data = 0
    if key == 32:  # SPACE to confirm
        bbox_data = ManualBBoxAnnotator().annotate(image_path)
    elif key == 27:  # ESC to skip
        cv2.destroyWindow(window_name)
        return None
    if bbox_data:
        return bbox_data
    else: return False
def main():
    files = ['./bbox_obv_Arcadius.json', 
            './bbox_obv_Constantine I.json', 
            './bbox_obv_Honorius.json',
            './bbox_obv_Theodosius I.json',
            './bbox_obv_Theodosius II.json',
            './bbox_obv_Valentinian III.json',
            './bbox_rev_rev.json']
    for file in files:
        with open(file, 'r') as f:
            file_data = json.load(f)
        for i in range(len(file_data)):
            box_metas = file_data[i]
            image_path, bbox = box_metas['filename'], box_metas['bbox']
            image_path = os.path.join('./images', image_path)
            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                continue
            exists = draw_bbox(image_path=image_path, list=bbox)
            if exists:
                box_metas['bbox'] = exists
                print(f"Updated bbox for {box_metas['filename']}")
        with open(file, 'w') as f:
            json.dump(file_data, f, indent=4)
        print(f"Saved changes to {file}")
if __name__ == "__main__":
    main()