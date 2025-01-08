import cv2
import mediapipe as mp
import pyautogui

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

            # Convert nose position to screen position
            screen_x = screen_w * nose_x
            screen_y = screen_h * nose_y

            # Use 8-axis movement logic
            move_x, move_y, direction = get_8axis_movement(nose_x, nose_y, screen_w, screen_h)

            # Optional: Move the cursor (can modify the speed of movement)
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
