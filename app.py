from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
import cv2
from fer import FER

app = Flask(__name__)
socketio = SocketIO(app)

# Initialize the FER detector
detector = FER()

# Open the webcam
video_capture = cv2.VideoCapture(0)

@app.route('/')
def index():
    return render_template('index.html')

def detect_emotion(frame):
    # Detect emotion
    emotion, score = detector.top_emotion(frame)
    return emotion, score

def generate_frames():
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        # Convert frame to RGB (OpenCV uses BGR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect emotion from frame
        emotion, score = detect_emotion(frame_rgb)

        # Send the emotion to the frontend using WebSocket
        socketio.emit('emotion', {'emotion': emotion, 'score': score})

        # Convert the frame to JPEG to send it as an HTTP response
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    socketio.run(app, debug=True)
