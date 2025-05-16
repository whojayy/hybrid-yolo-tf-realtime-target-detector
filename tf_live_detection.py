import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import mse

# Load the model with custom objects
model_path = os.path.join('Tensorflow', 'workspace', 'keras_model', 'hand_gesture_model.h5')
custom_objects = {
    'mse': mse,
    'bbox_mse': mse  # If this is a custom metric, we're mapping it to the standard mse
}
model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

# Class mapping
CLASS_MAPPING = {
    0: 'ThumbsUp',
    1: 'ThumbsDown',
    2: 'ThankYou',
    3: 'LiveLong'
}

# Run live detection
def run_live_detection():
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    # Check if webcam opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    print("Press 'q' to quit")
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break
        
        # Preprocess the frame
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (224, 224))
        img_normalized = img_resized / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        # Perform prediction
        predictions = model.predict(img_batch, verbose=0)  # Suppress verbose output
        
        # Handle different model output formats
        if isinstance(predictions, list) and len(predictions) == 2:
            # Model returns [class_predictions, box_predictions]
            class_pred = predictions[0][0]
            box_pred = predictions[1][0]
        else:
            # Try to handle other output formats
            print("Warning: Unexpected model output format. Attempting to adapt...")
            if isinstance(predictions, np.ndarray) and predictions.shape[-1] >= 5:
                # Assume format is [x1, y1, x2, y2, class1, class2, ...]
                box_pred = predictions[0, :4]
                class_pred = predictions[0, 4:]
            else:
                print("Error: Could not interpret model output")
                break
        
        # Get predicted class
        predicted_class_id = np.argmax(class_pred)
        predicted_class = CLASS_MAPPING[predicted_class_id]
        confidence = class_pred[predicted_class_id]
        
        # Get predicted bounding box (normalized)
        xmin, ymin, xmax, ymax = box_pred
        
        # Convert normalized coordinates to pixel coordinates
        height, width = frame.shape[:2]
        xmin_px = max(0, int(xmin * width))
        ymin_px = max(0, int(ymin * height))
        xmax_px = min(width, int(xmax * width))
        ymax_px = min(height, int(ymax * height))
        
        # Draw bounding box and label
        cv2.rectangle(frame, (xmin_px, ymin_px), (xmax_px, ymax_px), (0, 255, 0), 2)
        
        # Create label with class name and confidence
        label = f"{predicted_class}: {confidence:.2f}"
        
        # Calculate text size and position
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (xmin_px, ymin_px - 20), (xmin_px + text_width, ymin_px), (0, 255, 0), -1)
        cv2.putText(frame, label, (xmin_px, ymin_px - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Display the frame
        cv2.imshow('TensorFlow Hand Gesture Detection', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_live_detection()