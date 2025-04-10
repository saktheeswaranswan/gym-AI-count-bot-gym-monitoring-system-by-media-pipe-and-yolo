import cv2
import mediapipe as mp
import numpy as np
import csv
import json
import time

# -------------------------------
# Initialization and Configurations
# -------------------------------

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

# Debounce threshold (in seconds) – count only if keypoint remains inside for at least THRESHOLD seconds.
THRESHOLD = 0.5  # 0.5 seconds

# Counters and state variables:
# Separate counts for palm (wrist) and for fingers (average of tips)
palm_count = 0
finger_count = 0

palm_inside = False
finger_inside = False

# Timestamps for when they last exited the red inner box:
palm_last_exit_time = time.time()
finger_last_exit_time = time.time()

# Data lists for pose data and for count events:
pose_data = []
pose_json = []
# Each count event will be saved with its timestamp and cumulative counts.
count_data = []  # will be saved to "count_data.csv"

def is_inside_box(point, box):
    """Return True if the (x, y) point is within the given rectangular box."""
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

# -------------------------------
# Main Loop for Processing Frames
# -------------------------------
with mp_holistic.Holistic(min_detection_confidence=0.5, 
                          min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        timestamp = current_time

        # Flip frame for mirror-view and get dimensions.
        frame = cv2.flip(frame, 1)
        image_height, image_width, _ = frame.shape

        # Convert color and process with Mediapipe.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_frame)

        # Draw the outer (green) and inner (red) boxes.
        cv2.rectangle(frame, (outer_box['x'], outer_box['y']),
                      (outer_box['x'] + outer_box['w'], outer_box['y'] + outer_box['h']),
                      (0, 255, 0), 2)
        cv2.rectangle(frame, (inner_box['x'], inner_box['y']),
                      (inner_box['x'] + inner_box['w'], inner_box['y'] + inner_box['h']),
                      (0, 0, 255), 2)

        # -------------------------------
        # Process full-body pose landmarks.
        # -------------------------------
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
                cv2.putText(frame, str(idx), (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
            pose_data.append(pose_row)
            pose_json.append(pose_row)

            # Draw an elliptical arc for the right elbow (landmarks: 12, 14, 16).
            try:
                a = [landmarks[12].x * image_width, landmarks[12].y * image_height]
                b = [landmarks[14].x * image_width, landmarks[14].y * image_height]
                c = [landmarks[16].x * image_width, landmarks[16].y * image_height]
                
                angle = calculate_angle(a, b, c)
                angle_start = np.degrees(np.arctan2(a[1] - b[1], a[0] - b[0])) % 360
                angle_end = np.degrees(np.arctan2(c[1] - b[1], c[0] - b[0])) % 360

                cv2.ellipse(frame, tuple(np.int32(b)), (30, 30), 0,
                            angle_start, angle_end, (0, 255, 255), 2)
                cv2.putText(frame, f'{int(angle)} deg', tuple(np.int32(b)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            except Exception as e:
                pass

            # Mark foot keypoints (if available) such as left/right ankles and heels.
            foot_indices = {'Left Ankle': 27, 'Right Ankle': 28,
                            'Left Heel': 29, 'Right Heel': 30}
            for label, idx in foot_indices.items():
                if idx < len(landmarks):
                    x, y = int(landmarks[idx].x * image_width), int(landmarks[idx].y * image_height)
                    cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
                    cv2.putText(frame, label, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # -------------------------------
        # Process Right Hand Landmarks for Counting
        # -------------------------------
        if results.right_hand_landmarks:
            hand_landmarks = results.right_hand_landmarks.landmark

            # Draw right-hand keypoints.
            for idx, lm in enumerate(hand_landmarks):
                x = int(lm.x * image_width)
                y = int(lm.y * image_height)
                cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)
                cv2.putText(frame, str(idx), (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

            # --------- Compute Palm (Wrist) Point ---------
            if len(hand_landmarks) > 0:
                # Use index 0 (wrist) as the palm representative.
                wrist = hand_landmarks[0]
                palm_point = (int(wrist.x * image_width), int(wrist.y * image_height))
                cv2.circle(frame, palm_point, 5, (0, 255, 0), -1)

                # Debounce counting for palm:
                if is_inside_box(palm_point, inner_box):
                    # If palm just re-entered and has been outside for at least THRESHOLD seconds.
                    if not palm_inside and (current_time - palm_last_exit_time >= THRESHOLD):
                        palm_count += 1
                        palm_inside = True
                        # Log the event in count_data.
                        count_data.append({
                            'timestamp': timestamp,
                            'event': 'palm',
                            'palm_count': palm_count,
                            'finger_count': finger_count
                        })
                else:
                    # When the palm is not inside the inner box, update last_exit_time.
                    if palm_inside:
                        palm_last_exit_time = current_time
                    palm_inside = False

            # --------- Compute Fingers Average Point ---------
            # Use landmarks: thumb tip (4), index tip (8),
            # middle tip (12), ring tip (16), pinky tip (20).
            finger_key_indices = [4, 8, 12, 16, 20]
            valid_finger_points = []
            for i in finger_key_indices:
                if i < len(hand_landmarks):
                    valid_finger_points.append([hand_landmarks[i].x, hand_landmarks[i].y])
            if valid_finger_points:
                valid_finger_points = np.array(valid_finger_points)
                avg_finger_x = np.mean(valid_finger_points[:, 0]) * image_width
                avg_finger_y = np.mean(valid_finger_points[:, 1]) * image_height
                finger_point = (int(avg_finger_x), int(avg_finger_y))
                cv2.circle(frame, finger_point, 5, (0, 200, 0), -1)

                # Debounce counting for fingers:
                if is_inside_box(finger_point, inner_box):
                    if not finger_inside and (current_time - finger_last_exit_time >= THRESHOLD):
                        finger_count += 1
                        finger_inside = True
                        # Log the event.
                        count_data.append({
                            'timestamp': timestamp,
                            'event': 'fingers',
                            'palm_count': palm_count,
                            'finger_count': finger_count
                        })
                else:
                    if finger_inside:
                        finger_last_exit_time = current_time
                    finger_inside = False

        # -------------------------------
        # Process Left Hand Landmarks (Optional Visualization)
        # -------------------------------
        if results.left_hand_landmarks:
            left_hand = results.left_hand_landmarks.landmark
            for idx, lm in enumerate(left_hand):
                x = int(lm.x * image_width)
                y = int(lm.y * image_height)
                cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)
                cv2.putText(frame, str(idx), (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)

        # -------------------------------
        # Display and Keyboard Controls
        # -------------------------------
        cv2.putText(frame, f"Palm: {palm_count}  Fingers: {finger_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Holistic Pose Estimation', frame)

        key = cv2.waitKey(10) & 0xFF
        if key == 27:  # ESC to exit
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

# -------------------------------
# Save Output Data to CSV and JSON Files
# -------------------------------
# Save pose data.
csv_file = 'pose_data.csv'
with open(csv_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=pose_data[0].keys())
    writer.writeheader()
    writer.writerows(pose_data)

json_file = 'pose_data.json'
with open(json_file, 'w') as f:
    json.dump(pose_json, f, indent=2)

# Save count events data.
count_csv_file = 'count_data.csv'
with open(count_csv_file, 'w', newline='') as f:
    fieldnames = ['timestamp', 'event', 'palm_count', 'finger_count']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(count_data)

print(f"Saved {len(pose_data)} frames to {csv_file} and {json_file}")
print(f"Saved {len(count_data)} count events to {count_csv_file}")

