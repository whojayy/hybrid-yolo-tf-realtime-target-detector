import cv2
from ultralytics import YOLO

def run_live_detection():
    # Load the trained model
    model = YOLO('runs/detect/hand_gestures4/weights/best.pt')
    
    # Open webcam
    cap = cv2.VideoCapture(0)  # 0 is usually the default webcam
    
    # Check if webcam opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Class names for display
    class_names = ['ThumbsUp', 'ThumbsDown', 'ThankYou', 'LiveLong']
    
    print("Press 'q' to quit")
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break
        
        # Perform detection
        results = model(frame)
        
        # Process results
        for result in results:
            boxes = result.boxes  # Boxes object for bbox outputs
            
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0]  # Get box coordinates in (x1, y1, x2, y2) format
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Get class and confidence
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Create label with class name and confidence
                label = f"{class_names[cls]}: {conf:.2f}"
                
                # Calculate text size and position
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + text_width, y1), (0, 255, 0), -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Display the frame
        cv2.imshow('YOLOv8 Hand Gesture Detection', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_live_detection()