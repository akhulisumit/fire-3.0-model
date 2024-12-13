from ultralytics import YOLO
import cvzone
import cv2
import math

# Load the YOLO model
model = YOLO('fire.pt')

# Define class names
classnames = ['fire']

# Open the default camera (webcam)
cap = cv2.VideoCapture(0)

# Set the camera resolution (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Check if frame is read correctly
    if not ret:
        print("Failed to grab frame")
        break
    
    # Get frame dimensions
    frame_height, frame_width = frame.shape[:2]
    
    # Run inference on the frame
    result = model(frame, stream=True)
    
    # Process detected objects
    for info in result:
        boxes = info.boxes
        for box in boxes:
            # Calculate confidence
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            
            # Get class
            Class = int(box.cls[0])
            
            # Draw bounding box if confidence is above 50%
            if confidence > 50:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Calculate additional coordinate information
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Calculate relative positions
                relative_x_start = x1 / frame_width
                relative_y_start = y1 / frame_height
                relative_x_end = x2 / frame_width
                relative_y_end = y2 / frame_height
                
                relative_center_x = center_x / frame_width
                relative_center_y = center_y / frame_height
                
                # Detailed coordinate information
                print("Detailed Fire Detection Coordinates:")
                print(f"Absolute Coordinates:")
                print(f"  Top-left (x1, y1): ({x1}, {y1})")
                print(f"  Bottom-right (x2, y2): ({x2}, {y2})")
                print(f"  Center (x, y): ({center_x}, {center_y})")
                
                print(f"\nRelative Coordinates (0-1 range):")
                print(f"  Top-left (x1, y1): ({relative_x_start:.4f}, {relative_y_start:.4f})")
                print(f"  Bottom-right (x2, y2): ({relative_x_end:.4f}, {relative_y_end:.4f})")
                print(f"  Center (x, y): ({relative_center_x:.4f}, {relative_center_y:.4f})")
                
                print(f"\nBounding Box Dimensions:")
                print(f"  Width: {x2 - x1} pixels")
                print(f"  Height: {y2 - y1} pixels")
                
                print(f"\nFrame Information:")
                print(f"  Frame Width: {frame_width} pixels")
                print(f"  Frame Height: {frame_height} pixels")
                
                print(f"\nConfidence: {confidence}%\n")
                
                # Draw rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                
                # Put text with detailed information
                info_text = (f'{classnames[Class]} {confidence}% '
                             f'Pos:({center_x},{center_y}) '
                             f'Size:{x2-x1}x{y2-y1}')
                
                cvzone.putTextRect(frame, 
                                   info_text, 
                                   [x1 + 8, y1 + 100],
                                   scale=1.5, 
                                   thickness=2)
    
    # Display the resulting frame
    cv2.imshow('Fire Detection', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy windows
cap.release()
cv2.destroyAllWindows()