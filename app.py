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

# Timing variables to control click frequency
left_click_time = 0
right_click_time = 0
click_cooldown = 1  # Cooldown time in seconds

# Variables for smooth mouse movement
prev_screen_x, prev_screen_y = screen_width / 2, screen_height / 2
smooth_factor = 0.2  # Smoothing factor for pointer movement

@app.route('/track')
def track_face():
    global left_click_time, right_click_time, prev_screen_x, prev_screen_y

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

            # Map nose position to screen coordinates
            screen_x = np.interp(nose_x, [0, frame_width], [0, screen_width])
            screen_y = np.interp(nose_y, [0, frame_height], [0, screen_height])

            # Smooth the mouse movement
            screen_x = prev_screen_x + smooth_factor * (screen_x - prev_screen_x)
            screen_y = prev_screen_y + smooth_factor * (screen_y - prev_screen_y)

            prev_screen_x, prev_screen_y = screen_x, screen_y

            # Move the mouse to the smoothed position
            pyautogui.moveTo(screen_x, screen_y)

            # Eye landmarks for blinking detection
            left_eye_top = face_landmarks.landmark[159]
            left_eye_bottom = face_landmarks.landmark[145]
            right_eye_top = face_landmarks.landmark[386]
            right_eye_bottom = face_landmarks.landmark[374]

            # Calculate the vertical distance between top and bottom of the eyes
            left_eye_distance = np.linalg.norm(np.array([left_eye_top.x, left_eye_top.y]) - np.array([left_eye_bottom.x, left_eye_bottom.y]))
            right_eye_distance = np.linalg.norm(np.array([right_eye_top.x, right_eye_top.y]) - np.array([right_eye_bottom.x, right_eye_bottom.y]))

            # Threshold for detecting closed eyes
            blink_threshold = 0.02

            current_time = time.time()

            # Detect left eye blink for left-click
            if left_eye_distance < blink_threshold:
                if current_time - left_click_time > click_cooldown:
                    pyautogui.click(button='left')
                    left_click_time = current_time

            # Detect right eye blink for right-click
            if right_eye_distance < blink_threshold:
                if current_time - right_click_time > click_cooldown:
                    pyautogui.click(button='right')
                    right_click_time = current_time

            # Draw the landmarks on the frame (optional)
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)

    # Encode the frame in JPEG format
    ret, buffer = cv2.imencode('.jpg', frame)
    frame = buffer.tobytes()

    return Response(frame, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
