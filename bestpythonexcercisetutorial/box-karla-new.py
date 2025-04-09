import cv2
import mediapipe as mp
import numpy as np
import csv
import json
import time

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Initialize video capture
cap = cv2.VideoCapture('power.mp4')

# Box parameters
inner_box = {'x': 200, 'y': 150, 'w': 100, 'h': 100}
outer_box = {'x': 180, 'y': 130, 'w': 140, 'h': 140}
scale_step = 10
move_step = 10
cross_count = 0
was_outside = True

# Data collection lists
pose_data = []
pose_json = []

# Utility functions
def is_inside_box(point, box):
    px, py = point
    return (box['x'] < px < box['x'] + box['w']) and (box['y'] < py < box['y'] + box['h'])

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    return angle if angle <= 180 else 360 - angle

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        image_height, image_width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_frame)

        # Draw the boxes
        cv2.rectangle(frame, (outer_box['x'], outer_box['y']),
                      (outer_box['x'] + outer_box['w'], outer_box['y'] + outer_box['h']), (0, 255, 0), 2)
        cv2.rectangle(frame, (inner_box['x'], inner_box['y']),
                      (inner_box['x'] + inner_box['w'], inner_box['y'] + inner_box['h']), (0, 0, 255), 2)

        timestamp = time.time()

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2)
            )

            landmarks = results.pose_landmarks.landmark
            pose_row = {'timestamp': timestamp}
            for idx, lm in enumerate(landmarks):
                x, y = int(lm.x * image_width), int(lm.y * image_height)
                pose_row[f'x{idx}'] = lm.x
                pose_row[f'y{idx}'] = lm.y
                pose_row[f'z{idx}'] = lm.z
                pose_row[f'v{idx}'] = lm.visibility
                cv2.putText(frame, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            pose_data.append(pose_row)
            pose_json.append(pose_row)

            # Example angle: Right Elbow (shoulder-elbow-wrist)
            a = [landmarks[12].x * image_width, landmarks[12].y * image_height]
            b = [landmarks[14].x * image_width, landmarks[14].y * image_height]
            c = [landmarks[16].x * image_width, landmarks[16].y * image_height]
            angle = calculate_angle(a, b, c)
            cv2.putText(frame, f'{int(angle)} deg', tuple(np.int32(b)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # Detect palm position (Right Hand)
        if results.right_hand_landmarks:
            hand_landmarks = results.right_hand_landmarks.landmark
            palm_x = int((hand_landmarks[0].x + hand_landmarks[5].x + hand_landmarks[17].x) / 3 * image_width)
            palm_y = int((hand_landmarks[0].y + hand_landmarks[5].y + hand_landmarks[17].y) / 3 * image_height)
            cv2.circle(frame, (palm_x, palm_y), 5, (255, 0, 0), -1)

            if is_inside_box((palm_x, palm_y), inner_box) and not is_inside_box((palm_x, palm_y), outer_box):
                if was_outside:
                    cross_count += 1
                    was_outside = False
            else:
                was_outside = True

        # Display info
        cv2.putText(frame, f"Cross Count: {cross_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Holistic Pose Estimation', frame)

        key = cv2.waitKey(10) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('s'):
            inner_box['w'] += scale_step
            inner_box['h'] += scale_step
        elif key == ord('a'):
            outer_box['w'] += scale_step
            outer_box['h'] += scale_step
        elif key == 82:  # Up
            inner_box['y'] -= move_step
            outer_box['y'] -= move_step
        elif key == 84:  # Down
            inner_box['y'] += move_step
            outer_box['y'] += move_step
        elif key == 81:  # Left
            inner_box['x'] -= move_step
            outer_box['x'] -= move_step
        elif key == 83:  # Right
            inner_box['x'] += move_step
            outer_box['x'] += move_step

cap.release()
cv2.destroyAllWindows()

# Save CSV
csv_file = 'pose_data.csv'
with open(csv_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=pose_data[0].keys())
    writer.writeheader()
    writer.writerows(pose_data)

# Save JSON
json_file = 'pose_data.json'
with open(json_file, 'w') as f:
    json.dump(pose_json, f, indent=2)

print(f"Saved {len(pose_data)} frames to {csv_file} and {json_file}")

