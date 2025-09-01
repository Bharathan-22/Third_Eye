# Import necessary libraries
import cv2  # OpenCV library for computer vision tasks
import numpy as np  # NumPy for numerical computations
from PIL import Image  # PIL (Python Imaging Library) for image processing
import os  # Operating system functionalities

# Path for face image database
path = 'Face_Dataset'

# Create LBPH (Local Binary Patterns Histograms) face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
# Load Haar cascade classifier for face detection
detector = cv2.CascadeClassifier("Models/haarcascade_frontalface_default.xml");

# Define a function to load images and their corresponding labels from a directory
def getImagesAndLabels(path):
    # Get list of image file paths in the directory
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    # Initialize lists to store face images and their corresponding labels
    faceSamples=[]
    ids = []

    # Iterate through each image file path
    for imagePath in imagePaths:
        # Open and convert the image to grayscale
        PIL_img = Image.open(imagePath).convert('L')
        # Convert the image to a numpy array
        img_numpy = np.array(PIL_img,'uint8')

        # Extract the label (ID) from the image filename
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        
        # Detect faces in the image using the Haar cascade classifier
        faces = detector.detectMultiScale(img_numpy)

        # Iterate through each detected face
        for (x,y,w,h) in faces:
            # Crop the face region from the image
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            # Append the corresponding label to the labels list
            ids.append(id)

    # Return the list of face images and their corresponding labels
    return faceSamples,ids

# Print a message indicating that training is in progress
print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")

# Load face images and labels
faces, ids = getImagesAndLabels(path)
# Train the LBPH face recognizer using the loaded images and labels
recognizer.train(faces, np.array(ids))

# Save the trained model to 'Models/trainer.yml' file
recognizer.write('Models/trainer.yml') 

# Print the number of unique faces trained and exit the program
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
