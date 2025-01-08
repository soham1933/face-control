import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize webcam and face mesh
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()

# Define a function for 8-axis movement
def get_8axis_movement(nose_x, nose_y, screen_w, screen_h):
    move_x, move_y = 0, 0
    sensitivity = 0.01  # Adjust sensitivity for smaller head movements

    # Horizontal movement
    if nose_x < 0.5 - sensitivity:
        move_x = -1  # Move left
    elif nose_x > 0.5 + sensitivity:
        move_x = 1   # Move right

    # Vertical movement
    if nose_y < 0.5 - sensitivity:
        move_y = -1  # Move up
    elif nose_y > 0.5 + sensitivity:
        move_y = 1   # Move down

    # Diagonal movement combinations
    if move_x == -1 and move_y == -1:
        direction = "Up-Left"
    elif move_x == 1 and move_y == -1:
        direction = "Up-Right"
    elif move_x == -1 and move_y == 1:
        direction = "Down-Left"
    elif move_x == 1 and move_y == 1:
        direction = "Down-Right"
    elif move_x == -1:
        direction = "Left"
    elif move_x == 1:
        direction = "Right"
    elif move_y == -1:
        direction = "Up"
    elif move_y == 1:
        direction = "Down"
    else:
        direction = "Center"

    return move_x, move_y, direction

# Helper function to check if the mouth is open
def is_mouth_open(landmarks):
    upper_lip = landmarks[13].y
    lower_lip = landmarks[14].y
    return (lower_lip - upper_lip) > 0.03  # Adjust this threshold as needed

# Helper function to check if the left eye is closed
def is_left_eye_closed(landmarks):
    upper_left_eyelid = landmarks[159].y
    lower_left_eyelid = landmarks[145].y
    return (lower_left_eyelid - upper_left_eyelid) < 0.015  # Adjust the threshold

# Helper function to check if the right eye is closed
def is_right_eye_closed(landmarks):
    upper_right_eyelid = landmarks[386].y
    lower_right_eyelid = landmarks[374].y
    return (lower_right_eyelid - upper_right_eyelid) < 0.015  # Adjust the threshold

left_eye_closed_time = 0
right_eye_closed_time = 0
cursor_hold = False

try:
    while True:
        _, frame = cam.read()
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = face_mesh.process(rgb_frame)
        landmark_points = output.multi_face_landmarks
        frame_h, frame_w, _ = frame.shape

        if landmark_points:
            landmarks = landmark_points[0].landmark
            # Get the nose tip landmark (ID 1)
            nose = landmarks[1]
            nose_x = nose.x
            nose_y = nose.y

            # Check if the mouth is open
            if is_mouth_open(landmarks):
                cursor_hold = True  # Hold the cursor when mouth is open
            else:
                cursor_hold = False

            # Check if the left eye is closed
            if is_left_eye_closed(landmarks):
                if left_eye_closed_time == 0:
                    left_eye_closed_time = time.time()
                elif time.time() - left_eye_closed_time >= 3:
                    pyautogui.click(button='left')  # Left-click if closed for 3 seconds
                    left_eye_closed_time = 0
            else:
                left_eye_closed_time = 0

            # Check if the right eye is closed
            if is_right_eye_closed(landmarks):
                if right_eye_closed_time == 0:
                    right_eye_closed_time = time.time()
                elif time.time() - right_eye_closed_time >= 3:
                    pyautogui.click(button='right')  # Right-click if closed for 3 seconds
                    right_eye_closed_time = 0
            else:
                right_eye_closed_time = 0

            # Use 8-axis movement logic only when the cursor isn't held
            if not cursor_hold:
                move_x, move_y, direction = get_8axis_movement(nose_x, nose_y, screen_w, screen_h)
                if move_x or move_y:
                    pyautogui.moveRel(move_x * 20, move_y * 20)  # Adjust movement speed

            # Draw the nose point on the frame
            cv2.circle(frame, (int(nose.x * frame_w), int(nose.y * frame_h)), 3, (0, 0, 255), -1)

            # Show the direction of movement on the frame
            cv2.putText(frame, f'Movement: {direction}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Display the webcam feed with annotations
        cv2.imshow('Nose Controlled Mouse - 8 Axis', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("Program interrupted manually.")
finally:
    cam.release()
    cv2.destroyAllWindows()
