import cv2
import mediapipe as mp
import numpy as np
import csv
import json
import time

# Initialize MediaPipe Holistic and Drawing utilities
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Open video source
cap = cv2.VideoCapture('power.mp4')

# Box definitions:
#   outer_box (green): larger detection zone.
#   inner_box (red): scoring zone.
outer_box = {'x': 180, 'y': 130, 'w': 140, 'h': 140}  # Green box
inner_box = {'x': 200, 'y': 150, 'w': 100, 'h': 100}  # Red box

# Parameters for scaling and movement of boxes
scale_step = 10
move_step = 10

# Count updates only when the hand enters the red inner box after being completely outside the green outer box.
cross_count = 0
was_outside = True  # Flag is True when the hand was last entirely outside the green box.

# Data collection lists for pose data
pose_data = []
pose_json = []

def is_inside_box(point, box):
    """Return True if the (x, y) point is within the given box."""
    px, py = point
    return (box['x'] <= px <= box['x'] + box['w']) and (box['y'] <= py <= box['y'] + box['h'])

def calculate_angle(a, b, c):
    """
    Calculate the angle (in degrees) between vectors ab and cb with b as the vertex.
    """
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180 else 360 - angle

with mp_holistic.Holistic(min_detection_confidence=0.5, 
                          min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame for mirror-view
        frame = cv2.flip(frame, 1)
        image_height, image_width, _ = frame.shape

        # Process frame via MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_frame)

        # Draw the outer (green) and inner (red) boxes
        cv2.rectangle(frame, (outer_box['x'], outer_box['y']),
                      (outer_box['x'] + outer_box['w'], outer_box['y'] + outer_box['h']),
                      (0, 255, 0), 2)
        cv2.rectangle(frame, (inner_box['x'], inner_box['y']),
                      (inner_box['x'] + inner_box['w'], inner_box['y'] + inner_box['h']),
                      (0, 0, 255), 2)

        timestamp = time.time()

        # Process pose landmarks for full body visualization and data logging.
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2)
            )
            
            landmarks = results.pose_landmarks.landmark
            pose_row = {'timestamp': timestamp}
            for idx, lm in enumerate(landmarks):
                pose_row[f'x{idx}'] = lm.x
                pose_row[f'y{idx}'] = lm.y
                pose_row[f'z{idx}'] = lm.z
                pose_row[f'v{idx}'] = lm.visibility
                x, y = int(lm.x * image_width), int(lm.y * image_height)
                cv2.circle(frame, (x, y), 2, (255, 255, 255), -1)
                cv2.putText(frame, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

            pose_data.append(pose_row)
            pose_json.append(pose_row)

            # Draw an elliptical arc showing the right elbow angle using landmarks:
            #   right shoulder (12), right elbow (14), right wrist (16)
            try:
                a = [landmarks[12].x * image_width, landmarks[12].y * image_height]
                b = [landmarks[14].x * image_width, landmarks[14].y * image_height]
                c = [landmarks[16].x * image_width, landmarks[16].y * image_height]
                
                angle = calculate_angle(a, b, c)
                angle_start = np.degrees(np.arctan2(a[1] - b[1], a[0] - b[0])) % 360
                angle_end = np.degrees(np.arctan2(c[1] - b[1], c[0] - b[0])) % 360

                cv2.ellipse(frame, tuple(np.int32(b)), (30, 30), 0, angle_start, angle_end, (0, 255, 255), 2)
                cv2.putText(frame, f'{int(angle)} deg', tuple(np.int32(b)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            except Exception as e:
                pass

            # Mark foot keypoints (if available) such as left ankle (27) and right ankle (28),
            # left heel (29) and right heel (30)
            foot_indices = {'Left Ankle': 27, 'Right Ankle': 28, 'Left Heel': 29, 'Right Heel': 30}
            for label, idx in foot_indices.items():
                if idx < len(landmarks):
                    x, y = int(landmarks[idx].x * image_width), int(landmarks[idx].y * image_height)
                    cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
                    cv2.putText(frame, label, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # Process right hand landmarks
        if results.right_hand_landmarks:
            hand_landmarks = results.right_hand_landmarks.landmark

            # Draw right-hand keypoints with index labels
            for idx, lm in enumerate(hand_landmarks):
                x = int(lm.x * image_width)
                y = int(lm.y * image_height)
                cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)
                cv2.putText(frame, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

            # Compute a representative hand point using wrist and finger tip landmarks.
            # We use indices: 0 (wrist), 4 (thumb tip), 8 (index finger tip),
            # 12 (middle finger tip), 16 (ring finger tip), and 20 (pinky tip).
            key_indices = [0, 4, 8, 12, 16, 20]
            valid_points = []
            for i in key_indices:
                if i < len(hand_landmarks):
                    valid_points.append([hand_landmarks[i].x, hand_landmarks[i].y])
            if valid_points:
                valid_points = np.array(valid_points)
                avg_x = np.mean(valid_points[:, 0]) * image_width
                avg_y = np.mean(valid_points[:, 1]) * image_height
                hand_point = (int(avg_x), int(avg_y))
                cv2.circle(frame, hand_point, 5, (0, 255, 0), -1)

                # --- Revised Counting Logic ---
                # Update score only when the computed hand point is inside the red inner box
                # and the point was previously entirely outside the green outer box.
                if is_inside_box(hand_point, inner_box):
                    if was_outside:
                        cross_count += 1
                    was_outside = False
                else:
                    # Reset the flag only when the hand is completely out of the outer (green) box.
                    if not is_inside_box(hand_point, outer_box):
                        was_outside = True

        # Process left hand landmarks for completeness
        if results.left_hand_landmarks:
            left_hand = results.left_hand_landmarks.landmark
            for idx, lm in enumerate(left_hand):
                x = int(lm.x * image_width)
                y = int(lm.y * image_height)
                cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)
                cv2.putText(frame, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)

        # Display the score on the frame
        cv2.putText(frame, f"Score: {cross_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Holistic Pose Estimation', frame)

        # Key bindings: ESC to exit; 's' and 'a' adjust box sizes; arrow keys to move boxes.
        key = cv2.waitKey(10) & 0xFF
        if key == 27:  # ESC key
            break
        elif key == ord('s'):
            inner_box['w'] += scale_step
            inner_box['h'] += scale_step
        elif key == ord('a'):
            outer_box['w'] += scale_step
            outer_box['h'] += scale_step
        elif key == 82:  # Up arrow
            inner_box['y'] -= move_step
            outer_box['y'] -= move_step
        elif key == 84:  # Down arrow
            inner_box['y'] += move_step
            outer_box['y'] += move_step
        elif key == 81:  # Left arrow
            inner_box['x'] -= move_step
            outer_box['x'] -= move_step
        elif key == 83:  # Right arrow
            inner_box['x'] += move_step
            outer_box['x'] += move_step

cap.release()
cv2.destroyAllWindows()

# Save pose data to CSV and JSON files
csv_file = 'pose_data.csv'
with open(csv_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=pose_data[0].keys())
    writer.writeheader()
    writer.writerows(pose_data)

json_file = 'pose_data.json'
with open(json_file, 'w') as f:
    json.dump(pose_json, f, indent=2)

print(f"Saved {len(pose_data)} frames to {csv_file} and {json_file}")

