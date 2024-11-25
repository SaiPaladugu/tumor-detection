import os

# Define the base directory for the dataset
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'brain_tumor')

# Function to generate standardized filenames
def standardize_filename(index, prefix, extension):
    return f"{prefix}_{index:05d}.{extension}"  # Example: "train_00001.jpg"

def fix_naming_conventions(data_dir):
    # Iterate over "train" and "valid" directories
    for split in ['train', 'valid']:
        split_dir = os.path.join(data_dir, split)
        
        # Directories for images and labels
        images_dir = os.path.join(split_dir, 'images')
        labels_dir = os.path.join(split_dir, 'labels')

        # Ensure both directories exist
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            raise FileNotFoundError(f"Missing 'images' or 'labels' directory in {split_dir}")
        
        # Get image and label files
        image_files = {os.path.splitext(f)[0]: f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))}
        label_files = {os.path.splitext(f)[0]: f for f in os.listdir(labels_dir) if f.lower().endswith('.txt')}

        # Find matching pairs
        common_keys = set(image_files.keys()).intersection(label_files.keys())
        unmatched_images = set(image_files.keys()).difference(common_keys)
        unmatched_labels = set(label_files.keys()).difference(common_keys)

        # Delete unmatched images
        for unmatched_image in unmatched_images:
            file_to_delete = os.path.join(images_dir, image_files[unmatched_image])
            os.remove(file_to_delete)
            print(f"Deleted unmatched image: {file_to_delete}")
        
        # Delete unmatched labels
        for unmatched_label in unmatched_labels:
            file_to_delete = os.path.join(labels_dir, label_files[unmatched_label])
            os.remove(file_to_delete)
            print(f"Deleted unmatched label: {file_to_delete}")
        
        # Process matched files
        for index, key in enumerate(sorted(common_keys), start=1):
            # Old file paths
            old_image_path = os.path.join(images_dir, image_files[key])
            old_label_path = os.path.join(labels_dir, label_files[key])
            
            # New standardized names
            new_image_name = standardize_filename(index, split, "jpg")
            new_label_name = standardize_filename(index, split, "txt")
            new_image_path = os.path.join(images_dir, new_image_name)
            new_label_path = os.path.join(labels_dir, new_label_name)
            
            # Rename files
            os.rename(old_image_path, new_image_path)
            os.rename(old_label_path, new_label_path)
            
            print(f"Renamed: {old_image_path} -> {new_image_path}")
            print(f"Renamed: {old_label_path} -> {new_label_path}")

# Run the script
if __name__ == "__main__":
    # Ensure the data directory exists
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Data directory {DATA_DIR} does not exist.")
    
    fix_naming_conventions(DATA_DIR)
