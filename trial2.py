import cv2
import pytesseract
import sqlite3
import datetime

# Initialize the video capture
cap = cv2.VideoCapture(0)  # You can change this to a video file or camera IP

# Initialize your Haar Cascade Classifier for number plate detection
plate_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')  # Replace with your XML file
pytesseract.pytesseract.tesseract_cmd = r'D:\Tesseract-OCR\tesseract.exe'

# Initialize the SQLite database
conn = sqlite3.connect('number_plates.db')
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS number_plate (
        id INTEGER PRIMARY KEY,
        plate_text TEXT,
        capture_time DATETIME
    )
''')
conn.commit()

while True:
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

            # Save the detected plate and capture time in the database
            capture_time = datetime.datetime.now()
            cursor.execute('INSERT INTO number_plate (plate_text, capture_time) VALUES (?, ?)', (plate_text, capture_time))
            conn.commit()

            # Draw a bounding box around the detected plate
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame with number plate detection
    cv2.imshow('Number Plate Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close the database connection
cap.release()
conn.close()
cv2.destroyAllWindows()
