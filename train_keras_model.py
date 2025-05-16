import os
import tensorflow as tf
from tensorflow.keras import layers, models, applications, optimizers
import numpy as np
import cv2
import glob

# Define paths
WORKSPACE_PATH = os.path.join('Tensorflow', 'workspace')
TRAIN_PATH = os.path.join(WORKSPACE_PATH, 'yolo_dataset', 'train')
VAL_PATH = os.path.join(WORKSPACE_PATH, 'yolo_dataset', 'val')
MODEL_PATH = os.path.join(WORKSPACE_PATH, 'keras_model')

# Create model directory
os.makedirs(MODEL_PATH, exist_ok=True)

# Class mapping
CLASS_MAPPING = {
    0: 'ThumbsUp',
    1: 'ThumbsDown',
    2: 'ThankYou',
    3: 'LiveLong'
}

# Simplified load_dataset function
def load_dataset(images_dir, labels_dir, img_size=(224, 224)):
    images = []
    boxes = []
    classes = []
    
    # List all image files
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    print(f"Found {len(image_files)} image files in {images_dir}")
    
    for img_file in image_files:
        # Get image path
        img_path = os.path.join(images_dir, img_file)
        
        # Get label path
        base_name = os.path.splitext(img_file)[0]
        label_path = os.path.join(labels_dir, f"{base_name}.txt")
        
        # Check if both files exist
        if not os.path.exists(label_path):
            print(f"Warning: No label file for {img_file}")
            continue
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue
        
        # Process image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, img_size)
        img_normalized = img_resized / 255.0
        
        # Read label
        with open(label_path, 'r') as f:
            label_content = f.read().strip()
            
        # Parse label
        parts = label_content.split()
        if len(parts) != 5:
            print(f"Warning: Invalid label format in {label_path}")
            continue
            
        class_id = int(parts[0])
        center_x = float(parts[1])
        center_y = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        
        # Convert to xmin, ymin, xmax, ymax
        xmin = max(0, center_x - width/2)
        ymin = max(0, center_y - height/2)
        xmax = min(1, center_x + width/2)
        ymax = min(1, center_y + height/2)
        
        # Store data
        images.append(img_normalized)
        boxes.append([xmin, ymin, xmax, ymax])
        classes.append(class_id)
    
    print(f"Successfully loaded {len(images)} images")
    return np.array(images), np.array(boxes), np.array(classes)

# Create a custom model
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

# Train the model
def train_model():
    # Load data
    print("Loading training data...")
    train_images, train_boxes, train_classes = load_dataset(
        os.path.join(TRAIN_PATH, 'images'),
        os.path.join(TRAIN_PATH, 'labels')
    )
    
    print("Loading validation data...")
    val_images, val_boxes, val_classes = load_dataset(
        os.path.join(VAL_PATH, 'images'),
        os.path.join(VAL_PATH, 'labels')
    )
    
    print(f"Training data: {len(train_images)} images")
    print(f"Validation data: {len(val_images)} images")
    
    if len(train_images) == 0 or len(val_images) == 0:
        print("Error: No images found. Cannot train model.")
        return None
    
    # Create model
    print("Creating model...")
    model = create_model()
    
    # Train the model
    print("Training model...")
    history = model.fit(
        train_images,
        {'classification': train_classes, 'bbox': train_boxes},
        validation_data=(
            val_images,
            {'classification': val_classes, 'bbox': val_boxes}
        ),
        epochs=50,
        batch_size=16
    )
    
    # Save the model
    model.save(os.path.join(MODEL_PATH, 'hand_gesture_model.h5'))
    print(f"Model saved to {os.path.join(MODEL_PATH, 'hand_gesture_model.h5')}")
    
    return model

if __name__ == "__main__":
    train_model()