import os
import random
import shutil
from glob import glob
import xml.etree.ElementTree as ET

# Define paths
WORKSPACE_PATH = os.path.join('Tensorflow', 'workspace')
IMAGES_PATH = os.path.join(WORKSPACE_PATH, 'images')
ANNOTATION_PATH = os.path.join(WORKSPACE_PATH, 'annotations')
TRAIN_PATH = os.path.join(IMAGES_PATH, 'train')
TEST_PATH = os.path.join(IMAGES_PATH, 'test')
VOC_ANNOTATIONS = os.path.join('Tensorflow', 'labelimg', 'task_2_annotations_2025_05_16_04_22_36_pascal voc 1.1', 'Annotations')

# Create directories if they don't exist
for path in [TRAIN_PATH, TEST_PATH, ANNOTATION_PATH]:
    if not os.path.exists(path):
        os.makedirs(path)

# Get all image paths from the collected images folders
image_paths = []
labels = ['livelong', 'thankyou', 'thumbsdown', 'thumbsup']

for label in labels:
    label_dir = os.path.join(IMAGES_PATH, 'collectedimages', label)
    if os.path.exists(label_dir):
        # Get all jpg files in this directory
        label_images = glob(os.path.join(label_dir, '*.jpg'))
        image_paths.extend(label_images)

# Shuffle the dataset
random.shuffle(image_paths)

# Split into train (80%) and test (20%)
train_size = int(0.8 * len(image_paths))
train_paths = image_paths[:train_size]
test_paths = image_paths[train_size:]

print(f"Total images: {len(image_paths)}")
print(f"Training images: {len(train_paths)}")
print(f"Testing images: {len(test_paths)}")

# Copy images and their annotations to train and test folders
def copy_data(image_paths, destination):
    for img_path in image_paths:
        # Get the filename without extension
        filename = os.path.basename(img_path)
        base_filename = os.path.splitext(filename)[0]
        
        # Copy the image
        shutil.copy(img_path, os.path.join(destination, filename))
        
        # Find and copy the corresponding XML annotation
        xml_path = os.path.join(VOC_ANNOTATIONS, f"{base_filename}.xml")
        if os.path.exists(xml_path):
            # Update the path in the XML file to match the new location
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Update the filename in the XML
            for filename_elem in root.iter('filename'):
                filename_elem.text = filename
                
            # Update the path in the XML
            for path_elem in root.iter('path'):
                path_elem.text = os.path.join(destination, filename)
                
            # Save the modified XML to the destination
            tree.write(os.path.join(destination, f"{base_filename}.xml"))
        else:
            print(f"Warning: No annotation found for {img_path}")

# Copy data to train and test folders
copy_data(train_paths, TRAIN_PATH)
copy_data(test_paths, TEST_PATH)

print("Train/test split completed successfully!")