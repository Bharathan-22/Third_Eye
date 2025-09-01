# Import the necessary modules
from tkinter import *  # Import all classes and functions from the Tkinter module
import os  # Import the os module for interacting with the operating system

# Define functions for various image recognition tasks

# Function to capture face datasets
def captureDataset():
    print("Capture Dataset button clicked")  # Print a message indicating the button click
    os.system('python 01_face_dataset.py')  # Execute the script '01_face_dataset.py'

# Function for face recognition
def faceRec():
    print("Face Recognition button clicked")  # Print a message indicating the button click
    os.system('python 03_face_recognition.py')  # Execute the script '03_face_recognition.py'

# Function for object detection
def OBJ():
    print("Object Detection button clicked")  # Print a message indicating the button click
    os.system('python 04_object.py')  # Execute the script '04_object.py'

# Function for camera operations
def cam():
    print("Camera button clicked")  # Print a message indicating the button click
    os.system('python 001_camera.py')  # Execute the script '05_camera.py'

# Function to create and configure the GUI
def start():
    global root  # Declare root as a global variable
    root = Tk()  # Create a Tkinter window
    root.title('Image Recognition')  # Set the window title
    canvas = Canvas(root, width=720, height=50, bg='yellow')  # Create a canvas widget
    canvas.grid(column=0, row=0)  # Place the canvas in the window

    # Create and configure labels for the GUI
    heading = Label(root, text="Image Recognition", fg="#FFA500", bg="#fff")  # Create a label for the heading
    heading.config(font=('calibri 25'))  # Configure the font for the heading label
    heading.place(relx=0.32, rely=0.45)  # Place the heading label in the window
    heading.grid(column=0, row=0)  # Place the heading label in the window

    headings = Label(root, text="Select any one of the option", fg="#bf00ff", bg="#fff")  # Create a label for the instructions
    headings.config(font=('calibri 25'))  # Configure the font for the instructions label
    headings.place(relx=0.32, rely=0.45)  # Place the instructions label in the window
    headings.grid(column=0, row=1)  # Place the instructions label in the window

    # Create and configure buttons for each image recognition task
    buttoncd = Button(root, text='Capture Dataset', command=captureDataset, bg="red", fg="yellow")  # Create a button for capturing datasets
    buttoncd.configure(width=102, height=2, activebackground="#33B5E5", relief=RAISED)  # Configure the button appearance and behavior
    buttoncd.grid(column=0, row=2)  # Place the button in the window

    buttonfr = Button(root, text='Face Recognition', command=faceRec, bg="red", fg="yellow")  # Create a button for face recognition
    buttonfr.configure(width=102, height=2, activebackground="#33B5E5", relief=RAISED)  # Configure the button appearance and behavior
    buttonfr.grid(column=0, row=3)  # Place the button in the window

    buttonobj = Button(root, text='Object Detection', command=OBJ, bg="red", fg="yellow")  # Create a button for object detection
    buttonobj.configure(width=102, height=2, activebackground="#33B5E5", relief=RAISED)  # Configure the button appearance and behavior
    buttonobj.grid(column=0, row=4)  # Place the button in the window


    buttonc = Button(root, text='Camera', command=cam, bg="red", fg="yellow")  # Create a button for camera operations
    buttonc.configure(width=102, height=2, activebackground="#33B5E5", relief=RAISED)  # Configure the button appearance and behavior
    buttonc.grid(column=0, row=6)  # Place the button in the window

    exit_button = Button(root, text="Exit", command=root.destroy)  # Create an exit button
    exit_button.configure(width=102, height=2, activebackground="#33B5E5", relief=RAISED)  # Configure the button appearance and behavior
    exit_button.grid(column=0, row=7)  # Place the button in the window

    root.mainloop()  # Run the Tkinter event loop

# Check if the script is executed directly
if __name__ == '__main__':
    print("Starting the GUI")  # Print a message indicating the GUI startup
    start()  # Call the start function to create the GUI
