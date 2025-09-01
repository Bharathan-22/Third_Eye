# Import the OpenCV library
import cv2

# Initialize video capture from the default camera (index 0)
cap = cv2.VideoCapture('http://192.168.137.9:81/stream')
#cap = cv2.VideoCapture(0)

# Set the video width to 640 pixels
cap.set(3, 640)

# Set the video height to 480 pixels
cap.set(4, 480)

# Start an infinite loop to continuously capture and display video frames
while(1):
    # Read a frame from the video capture object
    ret, frame = cap.read()
    #frame =cv2.flip(frame,0)

    # Check if a frame is successfully read
    if ret == True:
        # Display the current frame in a window named "frame"
        cv2.imshow('frame', frame)

        # Wait for a key event for 1 millisecond
        k = cv2.waitKey(1) & 0xff

        # Check if the ESC key (key code 27) is pressed
        if k == 27:
            # If ESC is pressed, break the loop and exit the program
            break
    else:
        # If no frame is successfully read, break the loop and exit the program
        break

# Release the video capture object, releasing the webcam
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
