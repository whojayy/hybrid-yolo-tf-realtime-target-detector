### Hybrid YOLOV8 TensorFlow Realtime Target Detector

A comprehensive computer vision system for real-time detection and classification of hand gestures using TensorFlow/Keras and YOLOv8.

![Video Demo](videos/demo.gif)



## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)

- [Gesture Classes](#gesture-classes)
- [Annotation Process with CVAT](#annotation-process-with-cvat)
- [Dataset Statistics](#dataset-statistics)



- [Models](#models)

- [TensorFlow/Keras Model](#tensorflowkeras-model)
- [YOLOv8 Model](#yolov8-model)
- [Model Comparison](#model-comparison)



- [Installation](#installation)
- [Usage](#usage)

- [Data Preparation](#data-preparation)
- [Training](#training)
- [Live Detection](#live-detection)



- [Results](#results)
- [Challenges and Solutions](#challenges-and-solutions)




## Overview

This project implements a real-time hand gesture recognition system capable of detecting and classifying four distinct hand gestures. The system uses computer vision and deep learning techniques to process webcam input and identify gestures with bounding boxes and class labels.

Two different approaches were implemented and compared:

1. A custom TensorFlow/Keras model using MobileNetV2 as a backbone
2. A YOLOv8 model fine-tuned on our custom dataset


## Features

- Real-time detection and classification of hand gestures
- Support for four distinct hand gestures: ThumbsUp, ThumbsDown, ThankYou, and LiveLong
- Bounding box localization around detected gestures
- Confidence score display for each detection
- FPS (Frames Per Second) monitoring for performance evaluation
- Cross-platform compatibility (tested on macOS and Linux)


## Dataset

### Gesture Classes

The system recognizes four hand gestures:

1. **ThumbsUp** - A closed fist with thumb pointing upward
2. **ThumbsDown** - A closed fist with thumb pointing downward
3. **ThankYou** - Both hands pressed together in a prayer-like position
4. **LiveLong** - The Vulcan salute (palm forward with fingers forming a V shape)


### Annotation Process with CVAT

We used Computer Vision Annotation Tool (CVAT) deployed via Docker for annotating our dataset:

![Data Annotation Img](videos/data_Annotation.jpeg)

#### CVAT Setup

1. **Docker Installation**:

```shellscript
# Install Docker
sudo apt-get update
sudo apt-get install docker.io docker-compose
```


2. **CVAT Deployment**:

```shellscript
# Clone CVAT repository
git clone https://github.com/opencv/cvat
cd cvat

# Deploy CVAT with docker-compose
docker-compose up -d
```


3. **Accessing CVAT**:

1. Navigate to `http://localhost:8080` in a web browser
2. Create a new account or use default credentials





#### Annotation Workflow

1. **Project Creation**:

1. Created a new project named "Hand Gesture Recognition"
2. Defined four labels: ThumbsUp, ThumbsDown, ThankYou, LiveLong



2. **Image Upload**:

1. Uploaded raw images captured in various lighting conditions and backgrounds
2. Organized images into task batches for efficient annotation



3. **Annotation Process**:

1. Used bounding box annotation tool to draw boxes around each hand gesture
2. Assigned appropriate class labels to each bounding box
3. Ensured consistent annotation across the dataset



4. **Quality Control**:

1. Reviewed annotations for accuracy and consistency
2. Corrected any mislabeled or improperly bounded gestures
3. Ensured all images had appropriate annotations



5. **Export Format**:

1. Exported annotations in PASCAL VOC 1.1 XML format
2. Each image has a corresponding XML file with bounding box coordinates and class labels





### Dataset Statistics

- **Total Images**: ~500 images across all classes
- **Class Distribution**:

- ThumbsUp: 125 images
- ThumbsDown: 125 images
- ThankYou: 125 images
- LiveLong: 125 images



- **Train/Test Split**: 80% training, 20% testing
- **Image Resolution**: Various (resized during preprocessing)
- **Annotation Format**: PASCAL VOC 1.1 XML and YOLO txt format


## Project Structure

```plaintext
TFOD/
├── Tensorflow/
│   └── workspace/
│       ├── annotations/
│       │   ├── label_map.pbtxt
│       │   ├── train.record
│       │   └── test.record
│       ├── images/
│       │   ├── train/
│       │   │   ├── image1.jpg
│       │   │   ├── image1.xml
│       │   │   └── ...
│       │   └── test/
│       │       ├── image2.jpg
│       │       ├── image2.xml
│       │       └── ...
│       ├── yolo_dataset/
│       │   ├── train/
│       │   │   ├── images/
│       │   │   └── labels/
│       │   └── val/
│       │       ├── images/
│       │       └── labels/
│       ├── models/
│       │   └── my_ssd_mobnet/
│       └── keras_model/
│           └── hand_gesture_model.h5
├── runs/
│   └── detect/
│       └── hand_gestures4/
│           └── weights/
│               └── best.pt
├── 1. Image Collection.ipynb
├── 2. Training and Detection.ipynb
├── prepare_data.py
├── generate_tfrecord.py
├── train_keras_model.py
├── train_yolo.py
├── live_detection.py
├── tf_live_detection.py
├── requirements.txt
└── README.md
```

## Models

### TensorFlow/Keras Model

We implemented a custom TensorFlow/Keras model using transfer learning with MobileNetV2 as the backbone:

#### Architecture

```python
def create_model(num_classes=4):
    # Use MobileNetV2 as the base model
    base_model = applications.MobileNetV2(
        weights='imagenet', 
        include_top=False, 
        input_shape=(224, 224, 3)
    )
    
    # Freeze the base model
    base_model.trainable = False
    
    # Create the model
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    # Classification head
    classification_output = layers.Dense(num_classes, activation='softmax', name='classification')(x)
    
    # Bounding box regression head
    bbox_output = layers.Dense(4, name='bbox')(x)
    
    # Create the model with multiple outputs
    model = models.Model(inputs=inputs, outputs=[classification_output, bbox_output])
    
    # Compile the model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss={
            'classification': 'sparse_categorical_crossentropy',
            'bbox': 'mse'
        },
        metrics={
            'classification': 'accuracy',
            'bbox': 'mse'
        }
    )
    
    return model
```

#### Training Configuration

- **Optimizer**: Adam with learning rate 0.001
- **Loss Functions**:

- Classification: Sparse Categorical Crossentropy
- Bounding Box: Mean Squared Error (MSE)



- **Metrics**:

- Classification: Accuracy
- Bounding Box: MSE



- **Epochs**: 50
- **Batch Size**: 16
- **Input Shape**: 224x224x3 (RGB)


#### Performance

- **Classification Accuracy**: 100% on validation data
- **Bounding Box MSE**: 0.25 on validation data
- **Live Detection Performance**: Poor (struggled with real-time webcam input)


#### Known Issues

1. **Model Loading Problems**:

1. Custom metrics serialization issues when loading the model
2. Error: `TypeError: Could not locate function 'mse'`
3. Workaround: Use custom objects dictionary when loading



2. **Domain Shift**:

1. Model performed well on validation data but poorly on webcam input
2. Likely due to differences in lighting, angle, and background



3. **Inference Speed**:

1. Slower than YOLOv8 for real-time detection





### YOLOv8 Model

We also implemented a YOLOv8 model fine-tuned on our custom dataset:

#### Architecture

YOLOv8 (You Only Look Once version 8) is a state-of-the-art object detection model that uses a single neural network to predict bounding boxes and class probabilities directly from full images in one evaluation.

We used the YOLOv8n (nano) variant, which is optimized for speed while maintaining good accuracy.

#### Training Configuration

```python
from ultralytics import YOLO

# Load a pre-trained YOLOv8n model
model = YOLO('yolov8n.pt')

# Train the model on our custom dataset
results = model.train(
    data=os.path.join(YOLO_DIR, 'dataset.yaml'),
    epochs=50,
    imgsz=640,
    batch=16,
    name='hand_gestures'
)
```

- **Base Model**: YOLOv8n pre-trained on COCO dataset
- **Epochs**: 50
- **Image Size**: 640x640
- **Batch Size**: 16
- **Optimizer**: Default YOLOv8 optimizer (SGD with cosine learning rate scheduler)
- **Data Augmentation**: Default YOLOv8 augmentation pipeline (mosaic, random affine, etc.)


#### Dataset Configuration (YAML)

```yaml
train: /absolute/path/to/Tensorflow/workspace/yolo_dataset/train/images
val: /absolute/path/to/Tensorflow/workspace/yolo_dataset/val/images

nc: 4
names: ['thumbsup', 'thumbsdown', 'thankyou', 'livelong']
```

#### Performance

- **mAP@0.5**: 0.95 (mean Average Precision at IoU threshold 0.5)
- **mAP@0.5:0.95**: 0.83 (mean Average Precision across IoU thresholds 0.5 to 0.95)
- **Live Detection Performance**: Excellent (robust real-time detection)
- **FPS**: ~20-30 FPS on standard CPU


### Core Models 

| Aspect | TensorFlow/Keras Model | YOLOv8 Model
|-----|-----|-----
| Architecture | MobileNetV2 + Custom Heads | YOLOv8n
| Training Approach | Transfer Learning | Fine-tuning
| Input Size | 224x224 | 640x640
| Training Time | ~1 hour | ~2 hours
| Model Size | ~14 MB | ~6 MB
| Validation Accuracy | 100% | 95% mAP@0.5
| Live Detection | Poor | Excellent
| Inference Speed | Slower | Faster
| Robustness | Less robust to lighting/angle changes | More robust


## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster training)


### Environment Setup

```shellscript
# Clone the repository
git clone https://github.com/whojayy/YOLO_Object_Detection


# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

The main dependencies are listed in `requirements.txt`:

```plaintext
tensorflow>=2.10.0
ultralytics>=8.0.0
opencv-python>=4.5.0
numpy>=1.20.0
pillow>=8.0.0
matplotlib>=3.5.0
```

## Usage

### Data Preparation

1. **Prepare the directory structure**:


```shellscript
python prepare_data.py
```

This script:

- Creates the necessary directory structure
- Splits the annotated data into training and testing sets (80/20 split)
- Copies images and XML annotations to the appropriate directories


2. **Generate TFRecord files** (for TensorFlow model):


```shellscript
python generate_tfrecord.py
```

This script:

- Creates a label map file mapping class names to IDs
- Converts XML annotations to TFRecord format
- Generates train.record and test.record files


3. **Prepare YOLO dataset** (for YOLOv8 model):


```shellscript
python reorganize_dataset.py
```

This script:

- Converts XML annotations to YOLO format (normalized coordinates in text files)
- Creates the YOLO directory structure with train and validation splits
- Generates a dataset.yaml configuration file


### Training

#### TensorFlow/Keras Model

```shellscript
python train_keras_model.py
```

This script:

- Loads and preprocesses the training and validation data
- Creates a custom model with MobileNetV2 backbone
- Trains the model for 50 epochs
- Saves the trained model to `Tensorflow/workspace/keras_model/hand_gesture_model.h5`


#### YOLOv8 Model

```shellscript
python train_yolo.py
```

This script:

- Loads a pre-trained YOLOv8n model
- Fine-tunes it on our custom dataset for 50 epochs
- Saves the trained model to `runs/detect/hand_gestures4/weights/best.pt`
- Generates training metrics and visualizations


### Live Detection

#### TensorFlow/Keras Model

```shellscript
python tf_live_detection.py
```

This script:

- Loads the trained TensorFlow model
- Captures frames from the webcam
- Processes each frame for hand gesture detection
- Displays the results with bounding boxes and labels


#### YOLOv8 Model (Recommended)

```shellscript
python live_detection.py
```

This script:

- Loads the trained YOLOv8 model
- Captures frames from the webcam
- Performs real-time detection of hand gestures
- Displays the results with bounding boxes, labels, and confidence scores
- Shows FPS information for performance monitoring


## Results

### TensorFlow/Keras Model

The TensorFlow/Keras model achieved excellent metrics during training:

- Classification accuracy: 100% on validation data
- Bounding box MSE: 0.25 on validation data


However, it performed poorly in live detection scenarios, likely due to:

- Overfitting to the training data
- Domain shift between training images and webcam input
- Issues with model loading and custom loss functions


### YOLOv8 Model

The YOLOv8 model demonstrated superior performance:

- mAP@0.5: 0.95 (mean Average Precision at IoU threshold 0.5)
- mAP@0.5:0.95: 0.83 (mean Average Precision across IoU thresholds)
- Excellent real-time detection capabilities
- Robust performance under different lighting conditions
- Fast inference speed suitable for real-time applications


## Challenges and Solutions

### 1. Data Format Conversion

**Challenge**: Converting annotations from XML to TFRecord and YOLO formats.

**Solution**: Created custom scripts to parse XML files and convert coordinates:

- For TFRecord: Implemented `parse_xml()` function in `generate_tfrecord.py`
- For YOLO: Implemented `convert_to_yolo_format()` function in `reorganize_dataset.py`


### 2. Model Loading Issues

**Challenge**: Errors when loading the TensorFlow model due to custom metrics.

**Solution**: Provided custom objects dictionary when loading the model:

```python
custom_objects = {
    'mse': tf.keras.losses.mse,
    'bbox_mse': tf.keras.losses.mse
}
model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
```

### 3. YOLOv8 Dataset Configuration

**Challenge**: "No labels found" warning during YOLOv8 training.

**Solution**: Reorganized the dataset to match YOLOv8's expected structure and created a proper YAML configuration:

```python
def create_yaml_config():
    yaml_content = f"""
train: {os.path.abspath(os.path.join(YOLO_DIR, 'train', 'images'))}
val: {os.path.abspath(os.path.join(YOLO_DIR, 'valid', 'images'))}

nc: 4
names: ['thumbsup', 'thumbsdown', 'thankyou', 'livelong']
"""
    
    with open(os.path.join(YOLO_DIR, 'dataset.yaml'), 'w') as f:
        f.write(yaml_content)
```

### 4. Live Detection Performance

**Challenge**: Poor performance of TensorFlow model in live detection.

**Solution**: Switched to YOLOv8 model which performed better in real-time scenarios:

```python
def run_yolo_detection():
    # Load the YOLOv8 model
    model_path = "runs/detect/hand_gestures4/weights/best.pt"
    model = YOLO(model_path)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        
        # Perform prediction
        results = model(frame, conf=0.25)
        
        # Process results
        for result in results:
            # Visualize results on the frame
            annotated_frame = result.plot()
            
            # Display the annotated frame
            cv2.imshow("YOLOv8 Hand Gesture Detection", annotated_frame)
```


### Author: Jay Mewada