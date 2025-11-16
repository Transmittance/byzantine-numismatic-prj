import os
from PIL import Image

def resize_image(input_path, output_path, size=(300, 300)):
    with Image.open(input_path) as img:
        width, height = img.size
        

        if width > height:

            new_height = size[1]
            new_width = int(width * (size[1] / height))
        else:

            new_width = size[0]
            new_height = int(height * (size[0] / width))
        

        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        left = (new_width - size[0]) / 2
        top = (new_height - size[1]) / 2
        right = (new_width + size[0]) / 2
        bottom = (new_height + size[1]) / 2
        
        img = img.crop((left, top, right, bottom))
        img.save(output_path)

input_folder = "images"
output_folder = "images_300"

os.makedirs(output_folder, exist_ok=True)


for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        resize_image(input_path, output_path)
        print(f"Обработано: {filename}")

print("Все изображения уменьшены!")