import cv2
import mediapipe as mp
import numpy as np
import os
from datetime import datetime

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

# Set up video capture
cap = cv2.VideoCapture('power.mp4')  # Use 0 for webcam

# Red box parameters
red_box = {'x': 200, 'y': 150, 'w': 150, 'h': 150}
box_color = (0, 0, 255)
box_thickness = 2
move_step = 10

# Counting and hold timers
counts = {'palm': 0, 'fingers': 0, 'wrist': 0}
hold_timers = {'palm': None, 'fingers': None, 'wrist': None}
min_hold_duration = 1.0  # Seconds to be continuously inside

# Logging
log_columns = ['timestamp', 'event_type', 'position_x', 'position_y', 'box_x', 'box_y', 'box_w', 'box_h', 'palm_count', 'fingers_count', 'wrist_count']
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)
log_file = os.path.join(output_dir, 'hand_tracking_log.csv')
if not os.path.exists(log_file):
    with open(log_file, 'w') as f:
        f.write(','.join(log_columns) + '\n')

# Helper functions
def is_inside_box(point, box):
    x, y = point
    return box['x'] <= x <= box['x'] + box['w'] and box['y'] <= y <= box['y'] + box['h']

def log_event(name, point, box, counts):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    with open(log_file, 'a') as f:
        f.write(f"{timestamp},{name},{point[0]},{point[1]},{box['x']},{box['y']},{box['w']},{box['h']},{counts['palm']},{counts['fingers']},{counts['wrist']}\n")

# Main loop
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands, \
     mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        current_time = datetime.now()

        hand_results = hands.process(rgb_frame)
        pose_results = pose.process(rgb_frame)

        # Reset detection flags
        detected = {'palm': None, 'fingers': None, 'wrist': None}

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # Wrist
                wrist_lm = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                wrist_point = (int(wrist_lm.x * width), int(wrist_lm.y * height))
                if is_inside_box(wrist_point, red_box):
                    detected['wrist'] = wrist_point

                # Palm (approximated with wrist)
                if is_inside_box(wrist_point, red_box):
                    detected['palm'] = wrist_point

                # Fingers
                tips = [mp_hands.HandLandmark.THUMB_TIP,
                        mp_hands.HandLandmark.INDEX_FINGER_TIP,
                        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                        mp_hands.HandLandmark.RING_FINGER_TIP,
                        mp_hands.HandLandmark.PINKY_TIP]
                tip_points = [(int(hand_landmarks.landmark[tip].x * width),
                               int(hand_landmarks.landmark[tip].y * height)) for tip in tips]
                avg_finger_point = tuple(np.mean(tip_points, axis=0).astype(int))
                if is_inside_box(avg_finger_point, red_box):
                    detected['fingers'] = avg_finger_point

        # Check for continuous presence
        for key in ['palm', 'fingers', 'wrist']:
            if detected[key]:
                if hold_timers[key] is None:
                    hold_timers[key] = current_time
                elif (current_time - hold_timers[key]).total_seconds() >= min_hold_duration:
                    counts[key] += 1
                    log_event(key.capitalize(), detected[key], red_box, counts)
                    hold_timers[key] = None  # reset after logging
            else:
                hold_timers[key] = None  # reset timer if not inside

        # Draw pose landmarks
        if pose_results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_pose_landmarks_style()
            )

        # Draw red box and counters
        cv2.rectangle(frame, (red_box['x'], red_box['y']),
                      (red_box['x'] + red_box['w'], red_box['y'] + red_box['h']),
                      box_color, box_thickness)

        cv2.putText(frame, f"Palm Count: {counts['palm']}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Finger Count: {counts['fingers']}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Wrist Count: {counts['wrist']}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Hand + Pose Tracking", frame)

        key = cv2.waitKey(10) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 82:  # Up arrow
            red_box['y'] = max(0, red_box['y'] - move_step)
        elif key == 84:  # Down arrow
            red_box['y'] = min(height - red_box['h'], red_box['y'] + move_step)
        elif key == 81:  # Left arrow
            red_box['x'] = max(0, red_box['x'] - move_step)
        elif key == 83:  # Right arrow
            red_box['x'] = min(width - red_box['w'], red_box['x'] + move_step)

        if cv2.getWindowProperty("Hand + Pose Tracking", cv2.WND_PROP_VISIBLE) < 1:
            break

cap.release()
cv2.destroyAllWindows()

