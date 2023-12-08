import cv2
import pytesseract
import sqlite3
import datetime
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import io  # Import io for BytesIO

# Initialize the video capture
cap = cv2.VideoCapture(0)  # You can change this to a video file or camera IP

# Initialize your Haar Cascade Classifier for number plate detection
plate_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')
pytesseract.pytesseract.tesseract_cmd = r'D:\Tesseract-OCR\tesseract.exe'

# Initialize the SQLite database
conn = sqlite3.connect('number_plates.db')
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS number_plate_trial3 (
        id INTEGER PRIMARY KEY,
        plate_text TEXT,
        capture_time DATETIME,
        plate_image BLOB
    )
''')
conn.commit()

# Function to capture and process frames
def process_frame():
    ret, frame = cap.read()

    # Convert the frame to grayscale for number plate detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect number plates
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(50, 50))

    for (x, y, w, h) in plates:
        # Crop the detected number plate
        plate_img = frame[y:y + h, x:x + w]

        # Perform OCR on the number plate image
        plate_text = pytesseract.image_to_string(plate_img)

        if plate_text:
            print("Detected Plate:", plate_text)

            # Generate a valid filename based on the current date and time
            current_time = datetime.datetime.now()
            formatted_time = current_time.strftime('%Y-%m-%d_%H-%M-%S.%f')
            filename = formatted_time + '.png'  # Use underscores instead of spaces and colons

            # Save the image with the valid filename
            cv2.imwrite(filename, plate_img)

            # Store the detected plate and capture time in the database
            capture_time = current_time
            with open(filename, 'rb') as image_file:
                image_data = image_file.read()
            cursor.execute('INSERT INTO number_plate_trial3 (plate_text, capture_time, plate_image) VALUES (?, ?, ?)',
                           (plate_text, capture_time, image_data))
            conn.commit()

            # Draw a bounding box around the detected plate
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame with number plate detection
    cv2.imshow('Number Plate Detection', frame)

    # After processing the frame, call this function again after a delay
    root.after(10, process_frame)



# Function to update the list of detected plates in the GUI
def update_plate_list():
    cursor.execute('SELECT * FROM number_plate_trial3')
    plates = cursor.fetchall()

    # Clear the current list
    for row in plate_tree.get_children():
        plate_tree.delete(row)

    # Update the list with detected plates
    for plate in plates:
        # Convert the binary image data to a PhotoImage
        plate_img_bytes = plate[2]  # Ensure the data is in bytes format
        plate_img_pil = Image.open(plate_img_bytes)
        img_data = ImageTk.PhotoImage(plate_img_pil)

        plate_tree.insert('', 'end', values=(plate[1], img_data, plate[0]))

        # Keep a reference to the PhotoImage to prevent it from being garbage collected
        img_references.append(img_data)


# Create a GUI window
root = tk.Tk()
root.title("Number Plate Detection")

# Create a button to start the detection process
start_button = tk.Button(root, text="Start Detection", command=process_frame)
start_button.pack()

# Create a treeview to display detected plates
plate_tree = ttk.Treeview(root, columns=("Plate Text", "Image"))
plate_tree.heading("Plate Text", text="Plate Text")
plate_tree.heading("Image", text="Image")
plate_tree.pack()

# Create a button to update the list of detected plates
update_button = tk.Button(root, text="Update Plate List", command=update_plate_list)
update_button.pack()

# Function to show a message when the application is closed
def on_closing():
    if cap.isOpened():
        cap.release()
    conn.close()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

# List to keep references to PhotoImage objects
img_references = []

# Schedule the plate list update
update_plate_list()

root.mainloop()
