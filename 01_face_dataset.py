# Import necessary libraries
import cv2  # OpenCV library for computer vision tasks
import os  # Operating system functionalities
import time  # Library for time-related functions
from Models.audio import say  # Custom module for audio output

# Say "Dataset Capture" using the 'say' module
say("Dataset Capture")

# Initialize the camera (Webcam)
cam = cv2.VideoCapture('http://192.168.137.133:81/stream')
#cam = cv2.VideoCapture(0)


# Load the Haar cascade classifier for face detection
face_detector = cv2.CascadeClassifier('Models/haarcascade_frontalface_default.xml')

# Prompt user to enter a numeric face id
face_id = input('\n enter user id end press <return> ==>  ')

# Print a message indicating initialization of face capture
print("\n [INFO] Initializing face capture. Look at the camera and wait ...")

# Initialize individual sampling face count
count = 0

# Main loop for capturing face images
while True:
    # Read frame from the camera
    ret, img = cam.read()
    img = cv2.flip(img, 0)  # Flip video image vertically for better alignment
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
    faces = face_detector.detectMultiScale(gray, 1.1, 6)  # Detect faces in the grayscale image

    # Loop through detected faces
    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        count += 1

        # Save the captured image into the datasets folder
        cv2.imwrite("Face_Dataset/user." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h, x:x+w])

        # Display the image with the rectangle around the face
        cv2.imshow('image', img)
    
    # Check for key press (ESC to exit)
    k = cv2.waitKey(1) & 0xff
    if k == 27:  # If ESC is pressed
        break
    elif count >= 100:  # Take 20 face samples and stop video
        print("\n [INFO] Image Captured.....Thank you for your Patience")
        time.sleep(0.2)
        cam.release()  # Release the camera
        cv2.destroyAllWindows()  # Close all OpenCV windows
        print("\n [INFO] Please wait while training completes")
        os.system('python 02_face_training.py')  # Execute face training script
        break

# Print message indicating image capturing and training completion
print("\n [INFO] Capturing and Training.....Thank you for your Patience")
time.sleep(1)  # Wait for 1 second
