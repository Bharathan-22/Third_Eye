import numpy as np
import time
import cv2
import os
from Models.audio import say

# Initialize
say("Object detection")

# Model configuration
prototxt = "Models/deploy.prototxt.txt"
model = "Models/deploy.caffemodel"
confidence_threshold = 0.80

# Classes and colors
CLASSES = ["None", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "plant", "sheep",
        "sofa", "train", "monitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Load model
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# Initialize video capture
print("[INFO] starting video stream...")
cap = cv2.VideoCapture('http://192.168.137.9:81/stream')

#cap = cv2.VideoCapture(0)

# Set smaller resolution for ESP32-CAM
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize variables
img_text = ''
found = set()

while True:
    ret, frame = cap.read()
    if not ret:
        continue
        
    # Optimize frame size for faster processing
    #frame = cv2.resize(frame, (320, 320), interpolation=cv2.INTER_AREA)
    frame = cv2.flip(frame, 0)
    
    # Process frame
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
            0.007843, (300, 300), 127.5)

    # Detect objects
    net.setInput(blob)
    detections = net.forward()

    # Process detections
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > confidence_threshold:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw detection box
            label = f"{CLASSES[idx]}: {confidence * 100:.2f}%"
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                    COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
            
            # Handle speech output
            detected_class = format(CLASSES[idx])
            print(detected_class)

            if detected_class != str(img_text):
                img_text = detected_class
                if detected_class not in found:
                    say(detected_class)
                    found.add(detected_class)
            else:
                found.clear()

    # Display frame
    cv2.imshow("ESP32-CAM Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
