import cv2
from tkinter import messagebox
# Display a message reminding drivers to drive safely and follow traffic laws
messagebox.showinfo("Driver Safety Reminder", "Please drive safely, follow traffic laws, and fasten your seatbelt at all times.")
from tensorflow.keras.models import load_model
import numpy as np
import requests
import gps

# Load pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load pre-trained model for emotion detection
model = load_model('emotion_detection_model.h5')

# Define the emotions
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Simulated driving speed and road speed limit
driving_speed = 70 # mph
road_speed_limit = 60  # mph

# simulated normal heart rate for driver
normal_heart_rate = 75 # BPM

# simuated driver current heart rate
current_heart_rate = 80 # BPM

# Google Maps API key
google_maps_api_key = 'AIzaSyBw_1yxOEPRDYTiX1zVdpoyxihIR0CvWLk'

# Initialize GPS connection (Simulated)
#session = gps.gps("localhost", "2947")
#session.stream(gps.WATCH_ENABLE | gps.WATCH_NEWSTYLE)

# Function to create and display the warning message window
def show_warning():
    warning_window = tk.Toplevel()
    warning_window.title("Driver Safety Warning")
    warning_label = tk.Label(warning_window, text="Please drive Safely! Follow Traffic Laws and Fasten Seatbelt at all times.")
    warning_label.pack()
    close_button = tk.Button(warning_window, text="Close", command=warning_window.destroy)
    close_button.pack()

while True:
    # Read the frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame into grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the face of the driver in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop over each detected face
    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Extract the ROI (which is the face)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray_resized = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        # Normalize the ROI
        roi_gray_resized = roi_gray_resized / 255.0

        # Reshape the ROI for model input
        roi_input = np.expand_dims(roi_gray_resized, axis=0)
        roi_input = np.expand_dims(roi_input, axis=-1)

        # Predict emotion
        predicted_emotion = model.predict(roi_input)
        emotion_index = np.argmax(predicted_emotion)
        emotion = emotion_labels[emotion_index]

        # Prompt corresponding message based on detected emotion
        cv2.putText(frame, f'Emotion: {emotion}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Check for emotions that might impact driving and display a warning message
        if emotion in ['Angry', 'Disgust', 'Fear', 'Sad']:
            cv2.putText(frame, 'WARNING: Extreme Emotion Detected, please consider taking a break.', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Find the nearest resting area using Google Maps API
            url = f'https://maps.googleapis.com/maps/api/place/nearbysearch/json?location=50.86285,-0.08785446&radius=50000&type=rest_area&key={google_maps_api_key}'
            response = requests.get(url)
            data = response.json()
            if 'results' in data and data['results']:
                nearest_rest_area = data['results'][0]['name']
                cv2.putText(frame, f'Nearest Rest Area: {nearest_rest_area}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                cv2.putText(frame, 'No resting areas found nearby', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Check for speed limit violation and display warning message
    if driving_speed > road_speed_limit:
        cv2.putText(frame, f'Speed Limit Exceeded! ({driving_speed} mph > {road_speed_limit} mph)', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Simulate heart rate detection and compare with normal heart rate
    if current_heart_rate > normal_heart_rate + 20:  # If current heart rate is 20 BPM or more comapred to the normal heart rate
        cv2.putText(frame, 'WARNING: High Heart Rate Detected, consider taking a break.', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Driver Emotion Detector', frame)

    # Break the loop if the letter 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
