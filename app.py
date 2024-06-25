from flask import Flask, render_template, Response, stream_with_context, request, jsonify
import cv2
import dlib
import numpy as np
import pyaudio
import math
import json
import time
import queue

app = Flask(__name__)

# Create a queue for alerts
alert_queue = queue.Queue()

# Load configuration
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Load face detector model
def load_face_detector():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(config['face_predictor_path'])
    return detector, predictor

# Initialize system camera in videocapture object, 0 is the default camera
def initialize_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Could not open camera")
    return cap

# Get camera matrix
def get_camera_matrix(frame):
    size = frame.shape
    focal_length = size[1]
    center = (size[1] // 2, size[0] // 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )
    return camera_matrix

# detect voice
def detect_voice(stream, chunk):
    try:
        data = stream.read(chunk, exception_on_overflow=False)
        audio_data = np.frombuffer(data, dtype=np.int16)
        
        # Apply a simple noise reduction technique
        noise_threshold = config['noise_threshold']
        audio_data = np.where(np.abs(audio_data) < noise_threshold, 0, audio_data)
        
        # Add a small constant to avoid division by zero
        rms = np.sqrt(np.mean(audio_data ** 2) + 1e-6)
        spl = 20 * math.log10(rms / (2 ** 15) + 1e-6) + 94  # Adjusted calculation
        return max(spl, 0)  # Ensure non-negative SPL
    except IOError as e:
        if e.errno == pyaudio.paInputOverflowed:
            print("Input overflowed")
        else:
            raise
    return 0

# Convert rotation vector to euler angles
def rotation_vector_to_euler_angles(rotation_vector):
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = 0
    return np.rad2deg(np.array([x, y, z]))

# Get head pose
def get_head_pose(landmarks, camera_matrix, dist_coeffs):
    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0)
    ])
    
    image_points = np.array([
        (landmarks.part(30).x, landmarks.part(30).y),
        (landmarks.part(8).x, landmarks.part(8).y),
        (landmarks.part(36).x, landmarks.part(36).y),
        (landmarks.part(45).x, landmarks.part(45).y),
        (landmarks.part(48).x, landmarks.part(48).y),
        (landmarks.part(54).x, landmarks.part(54).y)
    ], dtype="double")
    
    success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
    if not success:
        return None
    
    return rotation_vector_to_euler_angles(rotation_vector)

# Add alert to queue
def add_alert(message):
    alert_queue.put(message)

# Main video feed generator
def gen_frames():
    detector, predictor = load_face_detector()
    cap = initialize_camera()
    
    ret, frame = cap.read()
    if not ret:
        raise IOError("Could not grab frame")
    
    camera_matrix = get_camera_matrix(frame)
    dist_coeffs = np.zeros((4, 1))
    
    infractions = 0
    head_turn_start_time = None
    last_alert_time = 0
    alert_cooldown = 5
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        current_time = time.time()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        
        for face in faces:
            landmarks = predictor(gray, face)
            euler_angles = get_head_pose(landmarks, camera_matrix, dist_coeffs)
            
            if euler_angles is not None:
                pitch, yaw, roll = euler_angles
                
                direction = "Facing: Center"
                if abs(yaw) > config['head_angle_threshold']:
                    direction = "Facing: Left" if yaw < 0 else "Facing: Right"
                    if head_turn_start_time is None:
                        head_turn_start_time = current_time
                    elif current_time - head_turn_start_time > config['head_turn_duration']:
                        if current_time - last_alert_time > alert_cooldown:
                            infractions += 1
                            add_alert(f"Please face forward. Infraction {infractions}/3 committed")
                            last_alert_time = current_time
                        head_turn_start_time = None
                else:
                    head_turn_start_time = None
                
                cv2.putText(frame, direction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Yaw: {yaw:.2f}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                
                # Draw head pose arrow
                nose_tip = (landmarks.part(30).x, landmarks.part(30).y)
                arrow_length = 100
                arrow_end = (int(nose_tip[0] + arrow_length * np.sin(np.deg2rad(yaw))),
                             int(nose_tip[1] - arrow_length * np.sin(np.deg2rad(pitch))))
                cv2.arrowedLine(frame, nose_tip, arrow_end, (0, 255, 0), 2)
        
        # Check for maximum infractions
        if infractions >= 3:
            add_alert("Maximum infractions reached. Exam terminated.")
            break
        
        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        # Yield the frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/alerts')
def sse_alerts():
    def event_stream():
        while True:
            if not alert_queue.empty():
                alert = alert_queue.get()
                yield f"data: {alert}\n\n"
            time.sleep(0.1)
    return Response(stream_with_context(event_stream()), content_type='text/event-stream')

@app.route('/process_audio', methods=['POST'])
def process_audio():
    audio_data = request.get_json()
    spl = process_audio_data(audio_data)
    if spl > config['voice_threshold']:
        add_alert(f"Please keep quiet. Voice detected (SPL: {spl:.2f} dB)")
    return jsonify({'spl': spl})

def process_audio_data(audio_data):
    # Implement your audio processing logic here
    # This is a placeholder implementation
    audio_array = np.array(audio_data)
    rms = np.sqrt(np.mean(audio_array**2))
    spl = 20 * np.log10(rms / 1e-5)  # Assuming reference pressure of 20 µPa
    return spl

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))