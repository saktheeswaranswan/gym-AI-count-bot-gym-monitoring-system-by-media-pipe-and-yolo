prev_landmarks = None
prev_time = None
line_position = 0  # Initial position of the control line

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    # Draw pose landmarks
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Extract landmarks
        landmarks = results.pose_landmarks.landmark
        current_time = cv2.getTickCount() / cv2.getTickFrequency()

        if prev_landmarks and prev_time:
            # Calculate velocity and acceleration vectors
            velocities = []
            accelerations = []
            for i, landmark in enumerate(landmarks):
                prev_landmark = prev_landmarks[i]
                dx = landmark.x - prev_landmark.x
                dy = landmark.y - prev_landmark.y
                dz = landmark.z - prev_landmark.z
                dt = current_time - prev_time
                velocity = math.sqrt(dx**2 + dy**2 + dz**2) / dt
                velocities.append(velocity)

                if prev_time:
                    acceleration = (velocity - prev_velocities[i]) / dt
                    accelerations.append(acceleration)

            # Calculate center of mass (simplified)
            # Assuming equal mass for each landmark
            x_coords = [lm.x for lm in landmarks]
            y_coords = [lm.y for lm in landmarks]
            z_coords = [lm.z for lm in landmarks]
            center_of_mass = (
                sum(x_coords) / len(x_coords),
                sum(y_coords) / len(y_coords),
                sum(z_coords) / len(z_coords)
            )

            # Calculate reaction force vector (simplified)
            # In a real scenario, this would require additional data
            reaction_force = np.array([sum(velocities), sum(accelerations), 0])

            # Display vectors and center of mass
            for i, landmark in enumerate(landmarks):
                cv2.putText(frame, f'Vel: {velocities[i]:.2f}', (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                if accelerations:
                    cv2.putText(frame, f'Acc: {accelerations[i]:.2f}', (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, f'COM: ({center_of_mass[0]:.2f}, {center_of_mass[1]:.2f}, {center_of_mass[2]:.2f})', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.putText(frame, f'Reaction Force: {reaction_force}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        prev_landmarks = landmarks
        prev_time = current_time

    # Display the frame
    cv2.imshow('Pose Estimation', frame)

    # Handle arrow key inputs to move the control line
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key to exit
        break
    elif key == 2490368:  # Up arrow key
        line_position -= 1
    elif key == 2621440:  # Down arrow key
        line_position += 1

    # Draw the control line
    cv2.line(frame, (0, line_position), (frame.shape[1], line_position), (0, 0, 255), 2)
    cv2.line(frame, (0, line_position + 20), (frame.shape[1], line_position + 20), (0, 255, 255), 2)

cap.release()
cv2.destroyAllWindows()

