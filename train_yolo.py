import os
from ultralytics import YOLO

# Define paths
WORKSPACE_PATH = os.path.join('Tensorflow', 'workspace')
TRAIN_PATH = os.path.join(WORKSPACE_PATH, 'images', 'train')
TEST_PATH = os.path.join(WORKSPACE_PATH, 'images', 'test')
ANNOTATION_PATH = os.path.join(WORKSPACE_PATH, 'annotations')

# Create a YAML configuration file for the dataset
def create_dataset_yaml():
    yaml_path = os.path.join(WORKSPACE_PATH, 'dataset.yaml')
    
    with open(yaml_path, 'w') as f:
        f.write(f"train: {TRAIN_PATH}\n")
        f.write(f"val: {TEST_PATH}\n")
        f.write("nc: 4\n")  # Number of classes
        f.write("names: ['ThumbsUp', 'ThumbsDown', 'ThankYou', 'LiveLong']\n")
    
    print(f"Dataset configuration created at {yaml_path}")
    return yaml_path

# Convert Pascal VOC annotations to YOLO format
def convert_annotations():
    import xml.etree.ElementTree as ET
    import glob
    
    # Class mapping
    class_mapping = {
        'thumbsup': 0,
        'thumbsdown': 1,
        'thankyou': 2,
        'livelong': 3
    }
    
    # Process train and test directories
    for data_dir in [TRAIN_PATH, TEST_PATH]:
        # Get all XML files
        xml_files = glob.glob(os.path.join(data_dir, '*.xml'))
        
        for xml_file in xml_files:
            # Parse XML
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Get image dimensions
            size = root.find('size')
            width = float(size.find('width').text)
            height = float(size.find('height').text)
            
            # Create YOLO annotation file
            txt_file = os.path.splitext(xml_file)[0] + '.txt'
            
            with open(txt_file, 'w') as f:
                for obj in root.findall('object'):
                    class_name = obj.find('name').text.lower()
                    
                    if class_name in class_mapping:
                        class_id = class_mapping[class_name]
                        
                        # Get bounding box coordinates
                        bbox = obj.find('bndbox')
                        xmin = float(bbox.find('xmin').text)
                        ymin = float(bbox.find('ymin').text)
                        xmax = float(bbox.find('xmax').text)
                        ymax = float(bbox.find('ymax').text)
                        
                        # Convert to YOLO format (center_x, center_y, width, height)
                        center_x = ((xmin + xmax) / 2) / width
                        center_y = ((ymin + ymax) / 2) / height
                        bbox_width = (xmax - xmin) / width
                        bbox_height = (ymax - ymin) / height
                        
                        # Write to file
                        f.write(f"{class_id} {center_x} {center_y} {bbox_width} {bbox_height}\n")
    
    print("Annotations converted to YOLO format")

# Train the model
def train_yolo():
    # Create dataset configuration
    dataset_yaml = create_dataset_yaml()
    
    # Convert annotations to YOLO format
    convert_annotations()
    
    # Initialize a new YOLO model from scratch
    model = YOLO('yolov8n.yaml')  # Create a new YOLOv8 nano model
    
    # Train the model
    results = model.train(
        data=dataset_yaml,
        epochs=100,
        imgsz=640,
        batch=16,
        name='hand_gestures'
    )
    
    print(f"Training completed. Model saved at {os.path.join('runs', 'detect', 'hand_gestures')}")

if __name__ == '__main__':
    train_yolo()