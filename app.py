from flask import Flask, render_template, Response, request, jsonify
import cv2
import dlib
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import queue
import json
import os

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

# Audio callback function
def audio_callback(indata, frames, time, status):
    volume_norm = np.linalg.norm(indata) * 10
    if volume_norm > config['voice_threshold']:
        add_alert(f"Please keep quiet. Voice detected (Volume: {volume_norm:.2f})")

# Video feed generator
def gen_frames():
    detector, predictor = load_face_detector()
    cap = initialize_camera()
    
    ret, frame = cap.read()
    if not ret:
        raise IOError("Could not grab frame")
    
    camera_matrix = get_camera_matrix(frame)
    dist_coeffs = np.zeros((4, 1))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        
        for face in faces:
            landmarks = predictor(gray, face)
            # Simplified: just show video
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
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
    return Response(event_stream(), content_type='text/event-stream')

@app.route('/process_audio', methods=['POST'])
def process_audio():
    audio_data = request.get_json()
    spl = process_audio_data(audio_data)
    if spl > config['voice_threshold']:
        add_alert(f"Please keep quiet. Voice detected (SPL: {spl:.2f} dB)")
    return jsonify({'spl': spl})

def process_audio_data(audio_data):
    audio_array = np.array(audio_data)
    rms = np.sqrt(np.mean(audio_array**2))
    spl = 20 * np.log10(rms / 1e-5)  # Assuming reference pressure of 20 µPa
    return spl

def add_alert(message):
    alert_queue.put(message)

if __name__ == '__main__':
    sd.InputStream(callback=audio_callback).start()
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))