import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils

# Screen dimensions
screen_width, screen_height = pyautogui.size()

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

# Sensitivity and movement speed for joystick-like control
sensitivity = 0.002  # Initial sensitivity
movement_speed = 10  # Initial movement speed

# Threshold to determine if mouth is open
mouth_open_threshold = 0.03  # You can adjust this threshold based on your testing

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
            # Get head landmarks for direction detection
            left_eye_inner = face_landmarks.landmark[133]  # Left inner eye
            right_eye_inner = face_landmarks.landmark[362]  # Right inner eye
            nose_tip = face_landmarks.landmark[1]  # Nose tip (reference point)

            # Get mouth landmarks for open/close detection
            upper_lip = face_landmarks.landmark[13]  # Upper lip
            lower_lip = face_landmarks.landmark[14]  # Lower lip

            # Calculate the distance between the upper and lower lip
            mouth_open_distance = abs(upper_lip.y - lower_lip.y)
            mouth_open = mouth_open_distance > mouth_open_threshold

            if not mouth_open:  # Only move mouse if mouth is closed
                head_tilt_x = (left_eye_inner.x - right_eye_inner.x) * frame_width
                head_tilt_y = (nose_tip.y - ((left_eye_inner.y + right_eye_inner.y) / 2)) * frame_height

                if abs(head_tilt_x) > sensitivity:
                    if head_tilt_x > 0:
                        pyautogui.moveRel(-movement_speed, 0)  # Move left
                    else:
                        pyautogui.moveRel(movement_speed, 0)  # Move right

                if abs(head_tilt_y) > sensitivity:
                    if head_tilt_y > 0:
                        pyautogui.moveRel(0, -movement_speed)  # Move up
                    else:
                        pyautogui.moveRel(0, movement_speed)  # Move down

    # Display the frame
    cv2.imshow('Head Joystick Control', frame)

    # Adjust sensitivity and speed using keyboard
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('+'):  # Increase sensitivity
        sensitivity += 0.0001
    elif key == ord('-'):  # Decrease sensitivity
        sensitivity = max(0.0001, sensitivity - 0.0001)
    elif key == ord('u'):  # Increase speed
        movement_speed += 1
    elif key == ord('d'):  # Decrease speed
        movement_speed = max(1, movement_speed - 1)

    print(f'Sensitivity: {sensitivity}, Movement Speed: {movement_speed}')

# Release resources
cap.release()
cv2.destroyAllWindows()
