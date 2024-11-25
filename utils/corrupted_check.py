from PIL import Image
import os

# Path to the image directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
image_dir = os.path.join(BASE_DIR, 'data', 'brain_tumor', 'train', 'images')

# Check each image for corruption
for image_file in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_file)
    try:
        with Image.open(image_path) as img:
            img.verify()  # Verify that the image is not corrupted
        print(f"Verified: {image_file}")
    except Exception as e:
        print(f"Corrupted: {image_file} - {e}")