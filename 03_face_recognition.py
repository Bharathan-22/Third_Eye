# Import necessary libraries
import cv2  # OpenCV library for computer vision tasks
from Models.audio import say  # Custom module for audio output

# Say "Facial Recognition" using the 'say' module
say("Facial Recognition")

# Create LBPH (Local Binary Patterns Histograms) face recognizer and load the trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('Models/trainer.yml')

# Path to the Haar cascade classifier for face detection
cascadePath = "Models/haarcascade_frontalface_default.xml"
# Create a cascade classifier for face detection
faceCascade = cv2.CascadeClassifier(cascadePath)

# Font for text overlay
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize id counter
id = 0

# Names related to ids: Example ==> name: id=1, etc.
names = ['None', 'bharathan', 'abdul','bharathan','bharathan','bharathan'] 

# Initialize and start realtime video capture
cam = cv2.VideoCapture('http://192.168.137.133:81/stream')
#cam = cv2.VideoCapture(0)
# Define minimum window size to be recognized as a face
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

# Initialize variables for text overlay
img_text = ''
found = set()

# Main loop for facial recognition
while True:

    ret, img = cam.read()
    img = cv2.flip(img, 0)  # Flip vertically for mirror effect

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = faceCascade.detectMultiScale(gray,
                                          scaleFactor=1.1,
                                          minNeighbors=6,
                                          minSize=(int(minW), int(minH)))

    for (x, y, w, h) in faces:
        # Draw rectangle around the detected face
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Predict the id of the detected face and calculate confidence
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        # Check if confidence is less than 100, "0" is a perfect match
        conf = round(100 - confidence)

        if conf > 30:
            img_text = names[id]
            confidence = "  {0}%".format(conf)
            
            # Speak the recognized name if it's not in the 'found' set
            if str(img_text) not in found:
                say(str(img_text))
                found.add(str(img_text))
        else:
            img_text = "unknown"
            found.clear()
            confidence = "  {0}%".format(conf)
        
        # Put recognized name and confidence level on the image
        cv2.putText(img, str(img_text), (x+5, y-5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x+5, y+h-5), font, 1, (255, 255, 0), 1)  
    
    # Display the image with facial recognition overlay
    cv2.imshow('camera', img) 

    # Press 'ESC' to exit video
    k = cv2.waitKey(1) & 0xff 
    if k == 27:
        break

# Clean up resources
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
