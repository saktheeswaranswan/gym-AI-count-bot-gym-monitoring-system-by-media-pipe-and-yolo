import cv2
import mediapipe as mp
import numpy as np
import csv
import json
import time

# Initialize MediaPipe Holistic and Drawing utilities
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Initialize video capture from file
cap = cv2.VideoCapture('power.mp4')

# Box parameters (inner box is completely inside outer box)
inner_box = {'x': 200, 'y': 150, 'w': 100, 'h': 100}
outer_box = {'x': 180, 'y': 130, 'w': 140, 'h': 140}
scale_step = 10
move_step = 10

# Counting transitions from outside outer_box to inside inner_box
cross_count = 0
was_outside = True  # indicates if the hand was last detected outside the outer_box

# Data collection lists for pose data
pose_data = []
pose_json = []

# Utility functions
def is_inside_box(point, box):
    """Check if a (x, y) point is inside a given rectangular box."""
    px, py = point
    return (box['x'] <= px <= box['x'] + box['w']) and (box['y'] <= py <= box['y'] + box['h'])

def calculate_angle(a, b, c):
    """
    Calculate the angle between three points: a, b, c with b as the vertex.
    The angle is returned in degrees.
    """
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180 else 360 - angle

with mp_holistic.Holistic(min_detection_confidence=0.5, 
                          min_tracking_confidence=0.5,
                          enable_segmentation=False) as holistic:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame for natural interaction
        frame = cv2.flip(frame, 1)
        image_height, image_width, _ = frame.shape

        # Convert image color space for Mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_frame)

        # Draw the outer and inner boxes
        cv2.rectangle(frame, (outer_box['x'], outer_box['y']),
                      (outer_box['x'] + outer_box['w'], outer_box['y'] + outer_box['h']), (0, 255, 0), 2)
        cv2.rectangle(frame, (inner_box['x'], inner_box['y']),
                      (inner_box['x'] + inner_box['w'], inner_box['y'] + inner_box['h']), (0, 0, 255), 2)

        timestamp = time.time()

        # Process pose landmarks and record data
        if results.pose_landmarks:
            # Draw holistic pose landmarks (joints and connections)
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2)
            )

            landmarks = results.pose_landmarks.landmark
            pose_row = {'timestamp': timestamp}
            for idx, lm in enumerate(landmarks):
                # Store normalized coordinates for CSV/JSON
                pose_row[f'x{idx}'] = lm.x
                pose_row[f'y{idx}'] = lm.y
                pose_row[f'z{idx}'] = lm.z
                pose_row[f'v{idx}'] = lm.visibility

                # Draw each landmark index at its corresponding pixel location
                x, y = int(lm.x * image_width), int(lm.y * image_height)
                cv2.circle(frame, (x, y), 2, (255, 255, 255), -1)
                cv2.putText(frame, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

            pose_data.append(pose_row)
            pose_json.append(pose_row)

            # ----- Draw an elliptical arc to show the angle of the right elbow -----
            # Use landmarks: right shoulder (12), right elbow (14), right wrist (16)
            try:
                a = [landmarks[12].x * image_width, landmarks[12].y * image_height]  # Shoulder
                b = [landmarks[14].x * image_width, landmarks[14].y * image_height]  # Elbow (vertex)
                c = [landmarks[16].x * image_width, landmarks[16].y * image_height]  # Wrist

                angle = calculate_angle(a, b, c)

                # Compute start and end angles for the arc (in degrees)
                angle_start = np.degrees(np.arctan2(a[1] - b[1], a[0] - b[0])) % 360
                angle_end = np.degrees(np.arctan2(c[1] - b[1], c[0] - b[0])) % 360

                # Draw the elliptical arc (using fixed axes lengths for clear visualization)
                # Note: cv2.ellipse draws an arc between startAngle and endAngle (measured anti-clockwise)
                cv2.ellipse(frame, tuple(np.int32(b)), (30, 30), 0, angle_start, angle_end, (0, 255, 255), 2)
                cv2.putText(frame, f'{int(angle)} deg', tuple(np.int32(b)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            except Exception as e:
                # In case any landmark is missing
                pass

            # ----- Mark feet key points explicitly -----
            # Check for common foot landmarks (using indices from Mediapipe Pose)
            # For example, left ankle (27), right ankle (28), left heel (29), right heel (30)
            foot_indices = {'Left Ankle': 27, 'Right Ankle': 28, 'Left Heel': 29, 'Right Heel': 30}
            for label, idx in foot_indices.items():
                if idx < len(landmarks):
                    x, y = int(landmarks[idx].x * image_width), int(landmarks[idx].y * image_height)
                    cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
                    cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # ----- Process right hand landmarks and count box transitions -----
        if results.right_hand_landmarks:
            hand_landmarks = results.right_hand_landmarks.landmark

            # Draw all right-hand key points
            for idx, lm in enumerate(hand_landmarks):
                x = int(lm.x * image_width)
                y = int(lm.y * image_height)
                cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)
                cv2.putText(frame, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

            # Compute a "palm point" by averaging positions of key landmarks (e.g., wrist and two key finger landmarks)
            palm_x = int((hand_landmarks[0].x + hand_landmarks[5].x + hand_landmarks[17].x) / 3 * image_width)
            palm_y = int((hand_landmarks[0].y + hand_landmarks[5].y + hand_landmarks[17].y) / 3 * image_height)
            cv2.circle(frame, (palm_x, palm_y), 5, (0, 255, 0), -1)

            # Count a transition when the palm is detected inside the inner box after being outside the outer box.
            if is_inside_box((palm_x, palm_y), inner_box):
                if was_outside:
                    cross_count += 1
                was_outside = False
            else:
                # Reset flag when the palm is entirely outside the outer_box
                if not is_inside_box((palm_x, palm_y), outer_box):
                    was_outside = True

        # ----- Process left hand landmarks for completeness -----
        if results.left_hand_landmarks:
            left_hand = results.left_hand_landmarks.landmark
            for idx, lm in enumerate(left_hand):
                x = int(lm.x * image_width)
                y = int(lm.y * image_height)
                cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)
                cv2.putText(frame, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)

        # Display the cross count on the frame
        cv2.putText(frame, f"Cross Count: {cross_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show the final frame
        cv2.imshow('Holistic Pose Estimation', frame)

        # Key bindings for exiting and modifying boxes
        key = cv2.waitKey(10) & 0xFF
        if key == 27:  # ESC key to exit
            break
        elif key == ord('s'):
            # Increase size of inner box
            inner_box['w'] += scale_step
            inner_box['h'] += scale_step
        elif key == ord('a'):
            # Increase size of outer box
            outer_box['w'] += scale_step
            outer_box['h'] += scale_step
        elif key == 82:  # Up arrow: move boxes upward
            inner_box['y'] -= move_step
            outer_box['y'] -= move_step
        elif key == 84:  # Down arrow: move boxes downward
            inner_box['y'] += move_step
            outer_box['y'] += move_step
        elif key == 81:  # Left arrow: move boxes leftward
            inner_box['x'] -= move_step
            outer_box['x'] -= move_step
        elif key == 83:  # Right arrow: move boxes rightward
            inner_box['x'] += move_step
            outer_box['x'] += move_step

# Release resources and close windows
cap.release()
cv2.destroyAllWindows()

# Save the recorded pose data as CSV and JSON
csv_file = 'pose_data.csv'
with open(csv_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=pose_data[0].keys())
    writer.writeheader()
    writer.writerows(pose_data)

json_file = 'pose_data.json'
with open(json_file, 'w') as f:
    json.dump(pose_json, f, indent=2)

print(f"Saved {len(pose_data)} frames to {csv_file} and {json_file}")

