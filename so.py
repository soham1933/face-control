from flask import Flask, Response, jsonify
import cv2
import numpy as np
import mediapipe as mp
import pyautogui

app = Flask(__name__)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# Screen dimensions
screen_width, screen_height = pyautogui.size()

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

@app.route('/track')
def track_face():
    ret, frame = cap.read()
    if not ret:
        return jsonify({"error": "Failed to capture frame from webcam"}), 500

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

            # Move the mouse to the mapped position
            pyautogui.moveTo(screen_x, screen_y)

            # Eye landmarks for blinking detection
            left_eye_top = face_landmarks.landmark[159]
            left_eye_bottom = face_landmarks.landmark[145]
            right_eye_top = face_landmarks.landmark[386]
            right_eye_bottom = face_landmarks.landmark[374]

            left_eye_aspect_ratio = np.linalg.norm([left_eye_top.x - left_eye_bottom.x, left_eye_top.y - left_eye_bottom.y])
            right_eye_aspect_ratio = np.linalg.norm([right_eye_top.x - right_eye_bottom.x, right_eye_top.y - right_eye_bottom.y])

            # Threshold for blink detection
            blink_threshold = 0.02

            # Check for blinks
            if left_eye_aspect_ratio < blink_threshold:
                pyautogui.click(button='left')
            if right_eye_aspect_ratio < blink_threshold:
                pyautogui.click(button='right')

            # Draw the landmarks on the frame (optional)
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)

    # Encode the frame in JPEG format
    ret, buffer = cv2.imencode('.jpg', frame)
    frame = buffer.tobytes()

    # Return the frame as a responsea
    return Response(frame, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
