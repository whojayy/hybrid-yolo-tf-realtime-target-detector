import os
import tensorflow as tf
import xml.etree.ElementTree as ET
from PIL import Image
import io

# Define paths
WORKSPACE_PATH = os.path.join('Tensorflow', 'workspace')
ANNOTATION_PATH = os.path.join(WORKSPACE_PATH, 'annotations')
TRAIN_PATH = os.path.join(WORKSPACE_PATH, 'images', 'train')
TEST_PATH = os.path.join(WORKSPACE_PATH, 'images', 'test')
LABELMAP_PATH = os.path.join(ANNOTATION_PATH, 'label_map.pbtxt')

# Helper functions to replace dataset_util
def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

# Create label map
def create_label_map():
    labels = [
        {'name': 'ThumbsUp', 'id': 1},
        {'name': 'ThumbsDown', 'id': 2},
        {'name': 'ThankYou', 'id': 3},
        {'name': 'LiveLong', 'id': 4}
    ]
    
    # Create label map dictionary
    label_map_dict = {}
    for label in labels:
        label_map_dict[label['name']] = label['id']
    
    # Write label map file
    with open(LABELMAP_PATH, 'w') as f:
        for label in labels:
            f.write('item {\n')
            f.write(f'  name: "{label["name"]}"\n')
            f.write(f'  id: {label["id"]}\n')
            f.write('}\n')
    
    print(f"Label map created at {LABELMAP_PATH}")
    return label_map_dict

# Create TFRecord function
def create_tf_example(example):
    # Read the image
    img_path = example['path']
    with tf.io.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    
    # Get image dimensions
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size
    
    filename = example['filename'].encode('utf8')
    image_format = b'jpg'
    
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []
    
    for box in example['boxes']:
        xmins.append(box['xmin'] / width)
        xmaxs.append(box['xmax'] / width)
        ymins.append(box['ymin'] / height)
        ymaxs.append(box['ymax'] / height)
        classes_text.append(box['class'].encode('utf8'))
        classes.append(box['class_id'])
    
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/filename': bytes_feature(filename),
        'image/source_id': bytes_feature(filename),
        'image/encoded': bytes_feature(encoded_jpg),
        'image/format': bytes_feature(image_format),
        'image/object/bbox/xmin': float_list_feature(xmins),
        'image/object/bbox/xmax': float_list_feature(xmaxs),
        'image/object/bbox/ymin': float_list_feature(ymins),
        'image/object/bbox/ymax': float_list_feature(ymaxs),
        'image/object/class/text': bytes_list_feature(classes_text),
        'image/object/class/label': int64_list_feature(classes),
    }))
    
    return tf_example

# Parse XML annotations
def parse_xml(xml_path, label_map_dict):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    boxes = []
    
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        if class_name.lower() == 'thumbsup':
            class_id = label_map_dict['ThumbsUp']
            class_name = 'ThumbsUp'
        elif class_name.lower() == 'thumbsdown':
            class_id = label_map_dict['ThumbsDown']
            class_name = 'ThumbsDown'
        elif class_name.lower() == 'thankyou':
            class_id = label_map_dict['ThankYou']
            class_name = 'ThankYou'
        elif class_name.lower() == 'livelong':
            class_id = label_map_dict['LiveLong']
            class_name = 'LiveLong'
        else:
            continue
        
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        
        boxes.append({
            'class': class_name,
            'class_id': class_id,
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax
        })
    
    return boxes

# Create TFRecords
def create_tfrecords(data_dir, output_path, label_map_dict):
    examples = []
    
    for img_file in os.listdir(data_dir):
        if img_file.endswith('.jpg'):
            img_path = os.path.join(data_dir, img_file)
            xml_file = os.path.splitext(img_file)[0] + '.xml'
            xml_path = os.path.join(data_dir, xml_file)
            
            if os.path.exists(xml_path):
                boxes = parse_xml(xml_path, label_map_dict)
                
                if boxes:  # Only add if there are valid boxes
                    examples.append({
                        'filename': img_file,
                        'path': img_path,
                        'boxes': boxes
                    })
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with tf.io.TFRecordWriter(output_path) as writer:
        for example in examples:
            tf_example = create_tf_example(example)
            writer.write(tf_example.SerializeToString())
    
    print(f'Successfully created TFRecord file: {output_path}')
    print(f'Processed {len(examples)} examples')

# Main execution
def main():
    # Create label map and get dictionary
    label_map_dict = create_label_map()
    
    # Create TFRecords
    train_output_path = os.path.join(ANNOTATION_PATH, 'train.record')
    test_output_path = os.path.join(ANNOTATION_PATH, 'test.record')
    
    create_tfrecords(TRAIN_PATH, train_output_path, label_map_dict)
    create_tfrecords(TEST_PATH, test_output_path, label_map_dict)

if __name__ == '__main__':
    main()