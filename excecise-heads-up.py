import cv2
import mediapipe as mp
import time
import numpy as np
import math

# ----------------- Helper Functions -----------------

def normalize_vector(v):
    """Return a normalized (unit) vector for v."""
    mag = math.sqrt(v[0]**2 + v[1]**2)
    if mag == 0:
        return (0, 0)
    return (v[0] / mag, v[1] / mag)

def draw_vector(img, start_point, vector, color, label=""):
    """
    Draw an arrow on img from start_point with given vector.
    Also, display text with the vector’s magnitude and angle.
    """
    scale = 1  # adjust drawing scale if needed
    end_point = (int(start_point[0] + vector[0] * scale), 
                 int(start_point[1] + vector[1] * scale))
    cv2.arrowedLine(img, start_point, end_point, color, 2, tipLength=0.3)
    mag = math.sqrt(vector[0]**2 + vector[1]**2)
    angle = math.degrees(math.atan2(-vector[1], vector[0]))  # note: y axis inversion
    text = f"{label} | Mag: {mag:.2f}, Ang: {angle:.1f}"
    cv2.putText(img, text, (start_point[0] + 10, start_point[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

def calculate_angle(a, b, c):
    """
    Calculate the angle at point b given three (x, y) coordinates.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def compute_center_of_mass(landmarks, width, height):
    """Compute the approximate center-of-mass as the average of all landmark positions."""
    xs, ys = [], []
    for lm in landmarks.landmark:
        xs.append(lm.x * width)
        ys.append(lm.y * height)
    return (int(np.mean(xs)), int(np.mean(ys))) if xs and ys else (0, 0)

# ----------------- Setup MediaPipe Holistic -----------------

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ----------------- Global Variables -----------------

# Central horizontal lines for zone-based event counting (thumb crossing)
line_y = 300           # central line's y coordinate (modifiable via arrow keys)
line_gap = 50          # gap between red (upper) and yellow (lower) lines
event_count = 0        # counts events for thumb crossing from yellow to red
was_in_yellow = False  # flag for thumb zone tracking

# For vector computations on the right index finger (landmark 8)
prev_time_finger = None
prev_finger_pos = None
prev_velocity_finger = None

# For reaction force on the right ankle (using pose landmarks)
prev_time_ankle = None
prev_ankle_pos = None
prev_velocity_ankle = None

# For moving box event: the box moves automatically from left to right.
box_x = 100            # initial x position of the box
box_y = 100            # fixed y position of the box
box_w, box_h = 200, 200  # width and height of the box
box_speed = 5          # horizontal speed in pixels per frame
box_event_count = 0    # counts events when the right-hand palm enters the box
was_in_box = False     # flag to prevent multiple counts while the hand remains in the box

# For head event counting: when the head comes up.
head_event_count = 0   # counts head-up events
was_head_up = False    # flag to track state change
head_threshold = 150   # y-coordinate threshold (in pixels) for head-up detection

# ----------------- Setup Video Capture & Writer -----------------

cap = cv2.VideoCapture(0)  # live webcam input
# Set mobile-friendly resolution: 840 x 640
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 840)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

# Setup VideoWriter to save output video in 1080p (1920 x 1080) MP4 format
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_size = (1920, 1080)
writer = cv2.VideoWriter('output.mp4', fourcc, fps, output_size)

# ----------------- Main Loop -----------------

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror the frame and convert from BGR to RGB
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_height, image_width, _ = frame.shape

    # Process frame with Holistic (pose and hand landmarks)
    results = holistic.process(frame_rgb)

    # Draw landmarks if available
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    # --- Compute Center-of-Mass (using pose landmarks) ---
    if results.pose_landmarks:
        center_of_mass = compute_center_of_mass(results.pose_landmarks, image_width, image_height)
        cv2.circle(frame, center_of_mass, 5, (255, 0, 255), -1)
        cv2.putText(frame, "Center of Mass", (center_of_mass[0] + 5, center_of_mass[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)

    # --- Head Event Counting (using the nose landmark) ---
    if results.pose_landmarks:
        nose = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE]
        nose_px = (int(nose.x * image_width), int(nose.y * image_height))
        cv2.circle(frame, nose_px, 5, (0, 255, 255), -1)
        # If nose (head) is above the defined threshold (i.e. y value is less than threshold), count as "head up".
        if nose_px[1] < head_threshold:
            if not was_head_up:
                head_event_count += 1
                was_head_up = True
        else:
            was_head_up = False
        cv2.putText(frame, f"Head Count: {head_event_count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    # --- Event Counting: Right Thumb Crossing from Yellow to Red ---
    if results.right_hand_landmarks:
        # Right thumb tip is landmark index 4
        right_thumb = results.right_hand_landmarks.landmark[4]
        r_thumb_px = (int(right_thumb.x * image_width), int(right_thumb.y * image_height))
        cv2.circle(frame, r_thumb_px, 5, (0, 0, 255), -1)
        red_line_y = int(line_y - line_gap/2)
        yellow_line_y = int(line_y + line_gap/2)
        if r_thumb_px[1] < red_line_y:
            current_zone = "red"
        elif r_thumb_px[1] > yellow_line_y:
            current_zone = "yellow"
        else:
            current_zone = "middle"
        if was_in_yellow and current_zone == "red":
            event_count += 1
            was_in_yellow = False
        if current_zone == "yellow":
            was_in_yellow = True
        cv2.putText(frame, f"Zone Count: {event_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # --- Moving Box Event Counting ---
    # Update box position: move horizontally and wrap around when exceeding frame width.
    box_x += box_speed
    if box_x > image_width:
        box_x = -box_w
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 255, 0), 2)
    cv2.putText(frame, f"Box Count: {box_event_count}", (box_x, box_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    if results.right_hand_landmarks:
        rh_landmarks = results.right_hand_landmarks.landmark
        # Estimate palm center from landmarks 0 (wrist), 5 (index MCP), and 17 (pinky MCP)
        palm_x = (rh_landmarks[0].x + rh_landmarks[5].x + rh_landmarks[17].x) / 3
        palm_y = (rh_landmarks[0].y + rh_landmarks[5].y + rh_landmarks[17].y) / 3
        palm_px = (int(palm_x * image_width), int(palm_y * image_height))
        cv2.circle(frame, palm_px, 5, (255, 0, 0), -1)
        if (box_x < palm_px[0] < box_x + box_w) and (box_y < palm_px[1] < box_y + box_h):
            if not was_in_box:
                box_event_count += 1
                was_in_box = True
        else:
            was_in_box = False

    # --- Vector Computations for Right Index Finger (landmark 8) ---
    if results.right_hand_landmarks:
        right_index = results.right_hand_landmarks.landmark[8]
        r_index_px = (int(right_index.x * image_width), int(right_index.y * image_height))
        cv2.circle(frame, r_index_px, 5, (0, 255, 0), -1)
        curr_time = time.time()
        dt_finger = curr_time - prev_time_finger if prev_time_finger is not None and (curr_time - prev_time_finger) > 0 else 1e-6
        if prev_finger_pos is not None:
            velocity_finger = ((r_index_px[0] - prev_finger_pos[0]) / dt_finger,
                               (r_index_px[1] - prev_finger_pos[1]) / dt_finger)
            norm_vel_finger = normalize_vector(velocity_finger)
            draw_vector(frame, r_index_px, (norm_vel_finger[0]*20, norm_vel_finger[1]*20),
                        (0, 255, 0), label="Finger Vel")
            if prev_velocity_finger is not None:
                acceleration_finger = ((velocity_finger[0] - prev_velocity_finger[0]) / dt_finger,
                                       (velocity_finger[1] - prev_velocity_finger[1]) / dt_finger)
                norm_acc_finger = normalize_vector(acceleration_finger)
                draw_vector(frame, r_index_px, (norm_acc_finger[0]*20, norm_acc_finger[1]*20),
                            (255, 0, 0), label="Finger Acc")
                ortho_vector = (-norm_vel_finger[1], norm_vel_finger[0])
                draw_vector(frame, r_index_px, (ortho_vector[0]*20, ortho_vector[1]*20),
                            (255, 255, 0), label="Finger Ortho")
                res_x = norm_vel_finger[0] + norm_acc_finger[0]
                res_y = norm_vel_finger[1] + norm_acc_finger[1]
                norm_resultant = normalize_vector((res_x, res_y))
                draw_vector(frame, r_index_px, (norm_resultant[0]*20, norm_resultant[1]*20),
                            (0, 255, 255), label="Resultant")
            prev_velocity_finger = velocity_finger
        prev_finger_pos = r_index_px
        prev_time_finger = curr_time

    # --- Vector Computations for Right Ankle (reaction force) ---
    if results.pose_landmarks:
        right_ankle = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ANKLE]
        r_ankle_px = (int(right_ankle.x * image_width), int(right_ankle.y * image_height))
        cv2.circle(frame, r_ankle_px, 5, (0, 128, 255), -1)
        curr_time_ankle = time.time()
        dt_ankle = curr_time_ankle - prev_time_ankle if prev_time_ankle is not None and (curr_time_ankle - prev_time_ankle) > 0 else 1e-6
        if prev_ankle_pos is not None:
            velocity_ankle = ((r_ankle_px[0] - prev_ankle_pos[0]) / dt_ankle,
                              (r_ankle_px[1] - prev_ankle_pos[1]) / dt_ankle)
            if prev_velocity_ankle is not None:
                acceleration_ankle = ((velocity_ankle[0] - prev_velocity_ankle[0]) / dt_ankle,
                                      (velocity_ankle[1] - prev_velocity_ankle[1]) / dt_ankle)
                norm_acc = normalize_vector(acceleration_ankle)
                reaction_force = (-norm_acc[0]*30, -norm_acc[1]*30)
                draw_vector(frame, r_ankle_px, reaction_force,
                            (0, 165, 255), label="Ankle React")
                prev_velocity_ankle = velocity_ankle
            else:
                prev_velocity_ankle = velocity_ankle
        prev_ankle_pos = r_ankle_px
        prev_time_ankle = curr_time_ankle

    # --- Joint Angle Calculation (example: Right Elbow) ---
    if results.pose_landmarks:
        try:
            r_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
            r_elbow = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW]
            r_wrist = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST]
            r_shoulder_px = (int(r_shoulder.x * image_width), int(r_shoulder.y * image_height))
            r_elbow_px = (int(r_elbow.x * image_width), int(r_elbow.y * image_height))
            r_wrist_px = (int(r_wrist.x * image_width), int(r_wrist.y * image_height))
            angle_r_elbow = calculate_angle(r_shoulder_px, r_elbow_px, r_wrist_px)
            cv2.putText(frame, f"R Elbow: {int(angle_r_elbow)}", r_elbow_px,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        except Exception as e:
            pass

    # --- Draw the central horizontal lines (red and yellow) ---
    red_line_y = int(line_y - line_gap/2)
    yellow_line_y = int(line_y + line_gap/2)
    cv2.line(frame, (0, red_line_y), (image_width, red_line_y), (0, 0, 255), 2)
    cv2.line(frame, (0, yellow_line_y), (image_width, yellow_line_y), (0, 255, 255), 2)
    cv2.putText(frame, "Use Arrow Up/Down to move lines", (10, image_height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # --- Keyboard Controls ---
    key = cv2.waitKey(1) & 0xFF
    if key in [82, 2490368, 65362]:  # Up arrow keys
        line_y -= 5
    elif key in [84, 2621440, 65364]:  # Down arrow keys
        line_y += 5
    elif key == 27:  # ESC to exit
        break

    # --- Display and Save Output Frame ---
    output_frame = cv2.resize(frame, output_size)
    writer.write(output_frame)
    cv2.imshow("Live Webcam Holistic Inference", frame)

cap.release()
writer.release()
cv2.destroyAllWindows()
holistic.close()

