from flask import Flask, Response
import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time

app = Flask(__name__)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# Screen dimensions
screen_width, screen_height = pyautogui.size()

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

# Variables for pointer smoothing and speed adjustment
prev_screen_x, prev_screen_y = screen_width / 2, screen_height / 2
smooth_factor = 0.2  # Smoothing factor, adjust between 0 (no smoothing) to 1 (full smoothing)
speed_multiplier = 1.0  # Speed multiplier for pointer movement, adjust to increase/decrease speed

@app.route('/track')
def track_face():
    global prev_screen_x, prev_screen_y

    start_time = time.time()  # Start time for debugging
    ret, frame = cap.read()
    if not ret:
        return "Failed to capture frame from webcam", 500

    # Flip the frame horizontally for natural interaction
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Face Mesh
    result = face_mesh.process(rgb_frame)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            # Nose tip (landmark 1) for mouse movement
            nose_tip = face_landmarks.landmark[1]
            nose_x = int(nose_tip.x * frame_width)
            nose_y = int(nose_tip.y * frame_height)

            # Map nose position to screen coordinates with speed multiplier
            screen_x = np.interp(nose_x, [0, frame_width], [0, screen_width])
            screen_y = np.interp(nose_y, [0, frame_height], [0, screen_height])

            # Apply speed multiplier
            screen_x = prev_screen_x + speed_multiplier * (screen_x - prev_screen_x)
            screen_y = prev_screen_y + speed_multiplier * (screen_y - prev_screen_y)

            # Smooth the mouse movement
            screen_x = prev_screen_x + smooth_factor * (screen_x - prev_screen_x)
            screen_y = prev_screen_y + smooth_factor * (screen_y - prev_screen_y)

            prev_screen_x, prev_screen_y = screen_x, screen_y

            # Move the mouse to the smoothed and scaled position
            pyautogui.moveTo(screen_x, screen_y)

            # Draw the landmarks on the frame (optional)
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)

    # Encode the frame in JPEG format
    ret, buffer = cv2.imencode('.jpg', frame)
    frame = buffer.tobytes()

    end_time = time.time()  # End time for debugging
    print(f"Frame processing time: {end_time - start_time:.4f} seconds")

    # Return the frame as a response
    return Response(frame, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=False)
