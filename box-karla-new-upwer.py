import cv2
import mediapipe as mp
import numpy as np
import json
import os
from datetime import datetime

# -------------------------------
# Initialization
# -------------------------------
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture('power.mp4')  # Replace with 0 for webcam

# Red box setup
red_box = {'x': 200, 'y': 150, 'w': 150, 'h': 150}
box_color = (0, 0, 255)
box_thickness = 2
move_step = 10

# Counters and state flags
palm_count = 0
finger_count = 0
wrist_count = 0
palm_inside = False
fingers_inside = False
wrist_inside = False

# Logging
log_data = []
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)
log_file = os.path.join(output_dir, 'hand_tracking_log.json')

# -------------------------------
# Helpers
# -------------------------------
def is_inside_box(point, box):
    x, y = point
    return box['x'] <= x <= box['x'] + box['w'] and box['y'] <= y <= box['y'] + box['h']

def log_event(name, point, box, palm_count, finger_count, wrist_count):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    event = {
        'timestamp': timestamp,
        'event_type': str(name),
        'position': {'x': int(point[0]), 'y': int(point[1])},
        'box_position': {
            'x': int(box['x']), 'y': int(box['y']),
            'w': int(box['w']), 'h': int(box['h'])
        },
        'palm_count': int(palm_count),
        'finger_count': int(finger_count),
        'wrist_count': int(wrist_count)
    }
    log_data.append(event)
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)

# -------------------------------
# Main Loop
# -------------------------------
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # Draw red box
        cv2.rectangle(frame, (red_box['x'], red_box['y']),
                      (red_box['x'] + red_box['w'], red_box['y'] + red_box['h']),
                      box_color, box_thickness)

        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                # Wrist
                wrist = hand.landmark[mp_hands.HandLandmark.WRIST]
                wrist_pt = (int(wrist.x * width), int(wrist.y * height))
                cv2.circle(frame, wrist_pt, 5, (255, 0, 0), -1)
                if is_inside_box(wrist_pt, red_box):
                    if not wrist_inside:
                        wrist_count += 1
                        wrist_inside = True
                        log_event("Wrist", wrist_pt, red_box, palm_count, finger_count, wrist_count)
                    cv2.putText(frame, f"W:{wrist_count}", wrist_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                else:
                    wrist_inside = False

                # Palm (same as wrist here)
                palm_pt = wrist_pt
                if is_inside_box(palm_pt, red_box):
                    if not palm_inside:
                        palm_count += 1
                        palm_inside = True
                        log_event("Palm", palm_pt, red_box, palm_count, finger_count, wrist_count)
                    cv2.putText(frame, f"P:{palm_count}", (palm_pt[0], palm_pt[1] + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    palm_inside = False

                # Fingers (average tip point)
                finger_ids = [mp_hands.HandLandmark.THUMB_TIP,
                              mp_hands.HandLandmark.INDEX_FINGER_TIP,
                              mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                              mp_hands.HandLandmark.RING_FINGER_TIP,
                              mp_hands.HandLandmark.PINKY_TIP]
                tips = [(int(hand.landmark[fid].x * width),
                         int(hand.landmark[fid].y * height)) for fid in finger_ids]
                avg_tip = tuple(np.mean(tips, axis=0).astype(int))
                cv2.circle(frame, avg_tip, 5, (0, 255, 255), -1)

                if is_inside_box(avg_tip, red_box):
                    if not fingers_inside:
                        finger_count += 1
                        fingers_inside = True
                        log_event("Fingers", avg_tip, red_box, palm_count, finger_count, wrist_count)
                    cv2.putText(frame, f"F:{finger_count}", avg_tip, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                else:
                    fingers_inside = False

        # Display counts
        cv2.putText(frame, f"Palm Count: {palm_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Finger Count: {finger_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Wrist Count: {wrist_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Hand Tracking", frame)

        key = cv2.waitKey(10) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('a') or key == 81:  # LEFT
            red_box['x'] = max(0, red_box['x'] - move_step)
        elif key == ord('d') or key == 83:  # RIGHT
            red_box['x'] = min(width - red_box['w'], red_box['x'] + move_step)
        elif key == ord('w') or key == 82:  # UP
            red_box['y'] = max(0, red_box['y'] - move_step)
        elif key == ord('s') or key == 84:  # DOWN
            red_box['y'] = min(height - red_box['h'], red_box['y'] + move_step)

# Cleanup
cap.release()
cv2.destroyAllWindows()

