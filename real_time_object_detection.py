import os
import warnings

# Suppress TensorFlow logging messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module='keras')

import cv2 # type: ignore
import numpy as np # type: ignore
import tensorflow as tf # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore
from tkinter import Tk, Button, Label, StringVar
from PIL import Image, ImageTk

# Step 5: Prediction Function
def predict_object_type(image_path):
    model = load_model('trained_model.h5')
    img = Image.open(image_path)
    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0  # Normalize the pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    class_names = {0: 'colorful', 1: 'black', 2: 'transparent'}
    conveyor_belts = ['Conveyor Belt C', 'Conveyor Belt A', 'Conveyor Belt B']
    
    return class_names[class_index], conveyor_belts[class_index]

# GUI Functionality
def open_camera():
    global cap, image_label, result_text
    cap = cv2.VideoCapture(0)  # Start video capture from the first camera

    # Create a window to show the camera feed
    window = Tk()
    window.title("Kursi - Object Identifier")

    # Initialize StringVar for result display
    result_text = StringVar()
    result_text.set("Click 'Click Image' to take a picture.")
    
    # Create a label to show results
    result_label = Label(window, textvariable=result_text, font=("Helvetica", 14))
    result_label.pack(pady=10)

    # Create a label to display the camera feed
    image_label = Label(window)
    image_label.pack()

    # Create a button to capture the image
    capture_button = Button(window, text="Click Image", command=capture_image, font=("Helvetica", 14))
    capture_button.pack(pady=20)

    # Update the camera feed continuously
    update_camera_feed(window)

    window.protocol("WM_DELETE_WINDOW", lambda: close_camera(window))
    window.mainloop()

def update_camera_feed(window):
    global cap, image_label
    ret, frame = cap.read()
    if ret:
        # Draw a green rectangle in the center of the frame
        height, width, _ = frame.shape
        start_point = (width // 2 - 100, height // 2 - 100)  # Top-left corner
        end_point = (width // 2 + 100, height // 2 + 100)    # Bottom-right corner
        color = (0, 255, 0)  # Green color in BGR
        thickness = 2       # Thickness of the rectangle
        cv2.rectangle(frame, start_point, end_point, color, thickness)

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img = img.resize((640, 480))  # Resize for the Tkinter window
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk

    image_label.after(10, lambda: update_camera_feed(window))  # Refresh the camera feed

def capture_image():
    global cap, image_label, result_text

    ret, frame = cap.read()
    if ret:
        # Define the dimensions of the green box
        box_size = 200  # Size of the green box (200x200 pixels)
        
        # Calculate cropping coordinates
        height, width, _ = frame.shape
        start_point = (width // 2 - box_size // 2, height // 2 - box_size // 2)
        end_point = (width // 2 + box_size // 2, height // 2 + box_size // 2)

        # Crop the image to the region inside the green box
        cropped_frame = frame[start_point[1]:end_point[1], start_point[0]:end_point[0]]

        # Save the cropped image
        save_image(cropped_frame)

        # Predict the object type using the cropped frame
        object_type, conveyor = predict_object_type('captured_image.png')

        # Update GUI with prediction
        result_text.set(f"Object: {object_type} | Send to: {conveyor}")

def save_image(cropped_frame):
    # Define the filename and path
    filename = "captured_image.png"
    
    # Save the cropped image to disk
    cv2.imwrite(filename, cropped_frame)
    print(f"Image saved as {filename}")

def close_camera(window):
    cap.release()
    cv2.destroyAllWindows()
    window.destroy()

if __name__ == "__main__":
    # Start the camera GUI
    open_camera()
