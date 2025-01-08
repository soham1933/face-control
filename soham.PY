import ctypes
import subprocess
import sys
import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# Screen dimensions
screen_width, screen_height = pyautogui.size()

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

# To track previous nose positions for relative movement
prev_nose_x, prev_nose_y = None, None

# Set a time limit to avoid double-clicks for blinks
blink_duration_threshold = 3  # seconds needed to trigger actions (e.g., open keyboard)
left_eye_closed_time = 0
right_eye_closed_time = 0

# Speed factor to adjust how fast the cursor moves based on nose movement
speed_factor = 2.5

# Disable pyautogui failsafe (optional)
pyautogui.FAILSAFE = False

# Function to check for admin rights and elevate privileges
def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

# Function to open the on-screen keyboard
def open_keyboard():
    try:
        subprocess.Popen("osk")  # Command for Windows OS
    except Exception as e:
        print(f"Error opening on-screen keyboard: {e}")

# If not running as admin, request elevation
if not is_admin():
    print("Requesting elevation...")
    ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, " ".join(sys.argv), None, 1)
    sys.exit()

# Continuous loop to capture video and control the mouse
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from webcam")
        break

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

            # If this is the first frame, initialize the previous nose position
            if prev_nose_x is None and prev_nose_y is None:
                prev_nose_x, prev_nose_y = nose_x, nose_y

            # Calculate the movement delta (difference) between current and previous nose positions
            delta_x = (nose_x - prev_nose_x) * speed_factor
            delta_y = (nose_y - prev_nose_y) * speed_factor

            # Move the mouse cursor relative to its current position
            current_mouse_x, current_mouse_y = pyautogui.position()
            pyautogui.moveTo(current_mouse_x + delta_x, current_mouse_y + delta_y)

            # Update previous nose position for the next frame
            prev_nose_x, prev_nose_y = nose_x, nose_y

            # Eye landmarks for blinking detection (left and right)
            left_eye_top = face_landmarks.landmark[159]
            left_eye_bottom = face_landmarks.landmark[145]
            right_eye_top = face_landmarks.landmark[386]
            right_eye_bottom = face_landmarks.landmark[374]

            left_eye_aspect_ratio = np.linalg.norm([left_eye_top.x - left_eye_bottom.x, left_eye_top.y - left_eye_bottom.y])
            right_eye_aspect_ratio = np.linalg.norm([right_eye_top.x - right_eye_bottom.x, right_eye_top.y - right_eye_bottom.y])

            # Threshold for blink detection
            blink_threshold = 0.02

            current_time = time.time()

            # Check if both eyes are closed
            if left_eye_aspect_ratio < blink_threshold and right_eye_aspect_ratio < blink_threshold:
                if left_eye_closed_time == 0 or right_eye_closed_time == 0:
                    left_eye_closed_time = current_time
                    right_eye_closed_time = current_time
                elif current_time - left_eye_closed_time >= blink_duration_threshold and current_time - right_eye_closed_time >= blink_duration_threshold:
                    open_keyboard()  # Open the on-screen keyboard
                    left_eye_closed_time = 0  # Reset the timer after opening the keyboard
                    right_eye_closed_time = 0  # Reset the timer after opening the keyboard
            else:
                left_eye_closed_time = 0  # Reset if the eyes are open
                right_eye_closed_time = 0  # Reset if the eyes are open

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
