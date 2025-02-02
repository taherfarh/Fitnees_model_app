from flask import Flask, Response
import cv2
import mediapipe as mp
import numpy as np
import math
import pygame

app = Flask(__name__)

# Initialize pygame mixer
pygame.mixer.init()
# Load sound
sound = pygame.mixer.Sound(r"C:\Users\taher farh\Downloads\fitnees_app\get api model\سااااررينا\سااااررينا\siren-alert-96052.mp3")

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    ab = a - b
    bc = c - b

    # Calculate the angle
    angle = np.arccos(np.clip(np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc)), -1.0, 1.0))
    return math.degrees(angle)

def hummer():
    left_counter = 0  # Counter for left arm
    right_counter = 0  # Counter for right arm
    left_state = None  # State for left arm
    right_state = None  # State for right arm
    cap = cv2.VideoCapture(0)
    sound_playing = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Define landmarks for both arms
            arm_sides = {
                'left': {
                    'shoulder': mp_pose.PoseLandmark.LEFT_SHOULDER,
                    'elbow': mp_pose.PoseLandmark.LEFT_ELBOW,
                    'wrist': mp_pose.PoseLandmark.LEFT_WRIST,
                    'hip': mp_pose.PoseLandmark.LEFT_HIP
                },
                'right': {
                    'shoulder': mp_pose.PoseLandmark.RIGHT_SHOULDER,
                    'elbow': mp_pose.PoseLandmark.RIGHT_ELBOW,
                    'wrist': mp_pose.PoseLandmark.RIGHT_WRIST,
                    'hip': mp_pose.PoseLandmark.RIGHT_HIP
                }
            }

            # Get coordinates for both shoulders and both hips
            left_shoulder = [
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
            ]
            right_shoulder = [
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
            ]
            left_hip = [
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
            ]
            right_hip = [
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
            ]

            # Draw a line between both shoulders
            cv2.line(
                image,
                (int(left_shoulder[0] * image.shape[1]), int(left_shoulder[1] * image.shape[0])),
                (int(right_shoulder[0] * image.shape[1]), int(right_shoulder[1] * image.shape[0])),
                (0, 255, 255),  # Color: yellow
                2
            )

            # Draw a line between both hips
            cv2.line(
                image,
                (int(left_hip[0] * image.shape[1]), int(left_hip[1] * image.shape[0])),
                (int(right_hip[0] * image.shape[1]), int(right_hip[1] * image.shape[0])),
                (0, 255, 255),  # Color: yellow
                2
            )

            # Initialize flags for both arms' angle violations
            arm_violated = {'left': False, 'right': False}

            for side, joints in arm_sides.items():
                # Get coordinates for each side
                shoulder = [
                    landmarks[joints['shoulder'].value].x,
                    landmarks[joints['shoulder'].value].y,
                ]
                elbow = [
                    landmarks[joints['elbow'].value].x,
                    landmarks[joints['elbow'].value].y,
                ]
                wrist = [
                    landmarks[joints['wrist'].value].x,
                    landmarks[joints['wrist'].value].y,
                ]
                hip = [
                    landmarks[joints['hip'].value].x,
                    landmarks[joints['hip'].value].y
                ]

                # Calculate angles
                elbow_angle = calculate_angle(shoulder, elbow, wrist)
                shoulder_angle = calculate_angle(hip, shoulder, elbow)

                # Draw connections for the arm
                arm_connections = [
                    (joints['shoulder'], joints['elbow']),
                    (joints['elbow'], joints['wrist'])
                ]
                torso_connections = [
                    (joints['hip'], joints['shoulder'])
                ]

                joint_positions = {
                    'Shoulder': [shoulder[0] * image.shape[1], shoulder[1] * image.shape[0]],
                    'Elbow': [elbow[0] * image.shape[1], elbow[1] * image.shape[0]],
                    'Wrist': [wrist[0] * image.shape[1], wrist[1] * image.shape[0]],
                    'Hip': [hip[0] * image.shape[1], hip[1] * image.shape[0]]
                }

                # Draw arm connections
                for connection in arm_connections:
                    start_idx = connection[0].value
                    end_idx = connection[1].value

                    start_point = landmarks[start_idx]
                    end_point = landmarks[end_idx]

                    start_coords = (int(start_point.x * image.shape[1]), int(start_point.y * image.shape[0]))
                    end_coords = (int(end_point.x * image.shape[1]), int(end_point.y * image.shape[0]))

                    cv2.line(image, start_coords, end_coords,  (0,255,0), 2)

                # Draw torso connections
                for connection in torso_connections:
                    start_idx = connection[0].value
                    end_idx = connection[1].value

                    start_point = landmarks[start_idx]
                    end_point = landmarks[end_idx]

                    start_coords = (int(start_point.x * image.shape[1]), int(start_point.y * image.shape[0]))
                    end_coords = (int(end_point.x * image.shape[1]), int(end_point.y * image.shape[0]))


                    cv2.line(image, start_coords, end_coords, (0,255,0), 2)  # Different color for torso

                # Draw joints
                for joint, position in joint_positions.items():
                    cv2.circle(image, (int(position[0]), int(position[1])), 7, (0, 0, 255), -1)

                # Display angles
                cv2.putText(
                    image,
                    f' {int(elbow_angle)}',
                    tuple(np.multiply(elbow, [image.shape[1], image.shape[0]]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                cv2.putText(
                    image,
                    f' {int(shoulder_angle)}',
                    tuple(np.multiply(shoulder, [image.shape[1], image.shape[0]]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                # Check if the angles are outside the desired range
                elbow_max = 180
                shoulder_max = 30
                sagittal_angle_threshold = 90
                shoulder_max_back = 25  # Maximum angle for shoulder extension backward
                elbow_min_back = 0  

                if elbow_angle > elbow_max or shoulder_angle >= shoulder_max:
                    arm_violated[side] = True
                if elbow_angle < elbow_min_back or shoulder_angle > shoulder_max_back:
                     arm_violated[side] = True

                # if elbow_angle < sagittal_angle_threshold or shoulder_angle > sagittal_angle_threshold:
                if shoulder_angle> sagittal_angle_threshold:
                    arm_violated[side] = True
                if not arm_violated['left'] and not arm_violated['right']:
                    if side == 'left':
                        if elbow_angle > 160:
                            left_state = 'down'
                        if elbow_angle < 30 and left_state == 'down':
                            left_state = 'up'
                            left_counter += 1
                            print(f'Left Counter: {left_counter}')
                    if side == 'right':
                        if elbow_angle > 160:
                            right_state = 'down'
                        if elbow_angle < 30 and right_state == 'down':
                            right_state = 'up'
                            right_counter += 1
                            print(f'Right Counter: {right_counter}')

            # Play sound if either arm is violated and not already playing
            if any(arm_violated.values()) and not sound_playing:
                sound.play()
                sound_playing = True
            elif not any(arm_violated.values()) and sound_playing:
                sound.stop()
                sound_playing = False

            # Draw counters on the image
            cv2.putText(image, f'Left Counter: {left_counter}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f'Right Counter: {right_counter}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Convert the image to JPEG format for streaming
        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()

        # Yield the frame to the Flask response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def dumbbell_front_raise():
    left_counter = 0
    right_counter = 0
    left_state = "down"
    right_state = "down"
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            arm_sides = {
                'left': {
                    'shoulder': mp_pose.PoseLandmark.LEFT_SHOULDER,
                    'elbow': mp_pose.PoseLandmark.LEFT_ELBOW,
                    'wrist': mp_pose.PoseLandmark.LEFT_WRIST,
                    'hip': mp_pose.PoseLandmark.LEFT_HIP
                },
                'right': {
                    'shoulder': mp_pose.PoseLandmark.RIGHT_SHOULDER,
                    'elbow': mp_pose.PoseLandmark.RIGHT_ELBOW,
                    'wrist': mp_pose.PoseLandmark.RIGHT_WRIST,
                    'hip': mp_pose.PoseLandmark.RIGHT_HIP
                }
            }

            for side, joints in arm_sides.items():
                shoulder = landmarks[joints['shoulder'].value]
                elbow = landmarks[joints['elbow'].value]
                wrist = landmarks[joints['wrist'].value]
                hip = landmarks[joints['hip'].value]

                # Get coordinates
                shoulder_coords = (int(shoulder.x * image.shape[1]), int(shoulder.y * image.shape[0]))
                elbow_coords = (int(elbow.x * image.shape[1]), int(elbow.y * image.shape[0]))
                wrist_coords = (int(wrist.x * image.shape[1]), int(wrist.y * image.shape[0]))
                hip_coords = (int(hip.x * image.shape[1]), int(hip.y * image.shape[0]))

                # Draw lines and points
                cv2.line(image, shoulder_coords, elbow_coords, (0, 255, 0), 2)
                cv2.line(image, elbow_coords, wrist_coords, (0, 255, 0), 2)
                cv2.line(image, hip_coords, shoulder_coords, (0, 255, 0), 2)

                for point in [shoulder_coords, elbow_coords, wrist_coords, hip_coords]:
                    cv2.circle(image, point, 7, (0, 0, 255), -1)

                # Calculate angles
                elbow_angle = calculate_angle([shoulder.x, shoulder.y], [elbow.x, elbow.y], [wrist.x, wrist.y])
                shoulder_angle = calculate_angle([hip.x, hip.y], [shoulder.x, shoulder.y], [elbow.x, elbow.y])

                # Display angles
                cv2.putText(image, f'{int(elbow_angle)}', elbow_coords, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(image, f'{int(shoulder_angle)}', shoulder_coords, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Check for correct form and count reps
                wrist_x = wrist.x * image.shape[1]
                shoulder_x = shoulder.x * image.shape[1]
                wrist_y = wrist.y * image.shape[0]
                shoulder_y = shoulder.y * image.shape[0]

                if side == 'left':
                    if wrist.y < shoulder.y and elbow_angle > 160 and left_state == "down":
                        # التأكد من أن اليد تكون للأمام وليس للجنب
                        if 30 < abs(wrist_x - shoulder_x) < 100:  # المسافة الأفقية بين المعصم والكتف
                            left_state = "up"
                            left_counter += 1
                    elif wrist.y > shoulder.y and elbow_angle > 160 and left_state == "up":
                        left_state = "down"
                elif side == 'right':
                    if wrist.y < shoulder.y and elbow_angle > 160 and right_state == "down":
                        # التأكد من أن اليد تكون للأمام وليس للجنب
                        if 30 < abs(wrist_x - shoulder_x) < 100:  # المسافة الأفقية بين المعصم والكتف
                            right_state = "up"
                            right_counter += 1
                    elif wrist.y > shoulder.y and elbow_angle > 160 and right_state == "up":
                        right_state = "down"

            # Display counters
            cv2.putText(image, f'Left Counter: {left_counter}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f'Right Counter: {right_counter}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)


        
        cv2.waitKey(1)  
        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



@app.route('/video_feed/<exercise>')
def video_feed(exercise):
    if exercise == 'hummer':
        return Response(hummer(), mimetype='multipart/x-mixed-replace; boundary=frame')
    elif exercise == 'front_raise':
        return Response(dumbbell_front_raise(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "Invalid exercise", 400

if __name__ == '__main__':
    app.run(debug=True)




@app.route('/api/pose_data')
def pose_data():
    # Get the pose data and return it as a JSON response
    data = hummer()
    return data

if __name__ == '__main__':
    app.run(debug=True)