import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# Screen dimensions
screen_width, screen_height = pyautogui.size()

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

# Variables to track the mouth state
mouth_open = False
prev_lip_x, prev_lip_y = pyautogui.position()

# Cooldown timer for eye blinks (to prevent multiple clicks)
last_left_blink_time = 0
last_right_blink_time = 0
blink_cooldown = 0.3  # 300 milliseconds cooldown between clicks

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
            # Lip landmarks for movement detection
            upper_lip = face_landmarks.landmark[13]  # Upper lip
            lower_lip = face_landmarks.landmark[14]  # Lower lip
            
            # Calculate the distance between the upper and lower lip to check if the mouth is open
            lip_distance = np.linalg.norm([upper_lip.x - lower_lip.x, upper_lip.y - lower_lip.y])

            # Check if the mouth is closed (based on a threshold for lip distance)
            mouth_open_threshold = 0.03
            if lip_distance < mouth_open_threshold:
                if mouth_open == False:  # If the mouth was open, now close it and continue movement
                    mouth_open = True

                # Move mouse when mouth is closed
                lip_x = int(upper_lip.x * frame_width)
                lip_y = int(upper_lip.y * frame_height)

                # Map lip position to screen coordinates
                screen_x = np.interp(lip_x, [0, frame_width], [0, screen_width])
                screen_y = np.interp(lip_y, [0, frame_height], [0, screen_height])

                # Smooth mouse movement
                smoothing_factor = 0.2
                screen_x = prev_lip_x + (screen_x - prev_lip_x) * smoothing_factor
                screen_y = prev_lip_y + (screen_y - prev_lip_y) * smoothing_factor

                # Move the mouse
                pyautogui.moveTo(screen_x, screen_y)

                # Update previous lip position
                prev_lip_x, prev_lip_y = screen_x, screen_y

            else:
                mouth_open = False  # Stop mouse movement when mouth is open

            # Eye blink detection for clicking
            left_eye_top = face_landmarks.landmark[159]
            left_eye_bottom = face_landmarks.landmark[145]
            right_eye_top = face_landmarks.landmark[386]
            right_eye_bottom = face_landmarks.landmark[374]

            left_eye_aspect_ratio = np.linalg.norm([left_eye_top.x - left_eye_bottom.x, left_eye_top.y - left_eye_bottom.y])
            right_eye_aspect_ratio = np.linalg.norm([right_eye_top.x - right_eye_bottom.x, right_eye_top.y - right_eye_bottom.y])

            # Blink threshold for eyes
            blink_threshold = 0.02
            current_time = time.time()

            # Check for left-eye blinks (left-click)
            if left_eye_aspect_ratio < blink_threshold and (current_time - last_left_blink_time) > blink_cooldown:
                pyautogui.click(button='left')
                last_left_blink_time = current_time

            # Check for right-eye blinks (right-click)
            if right_eye_aspect_ratio < blink_threshold and (current_time - last_right_blink_time) > blink_cooldown:
                pyautogui.click(button='right')
                last_right_blink_time = current_time

    # Display the frame
    cv2.imshow('Lip-Mouse Control', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
