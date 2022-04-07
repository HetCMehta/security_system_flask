from flask import Flask, render_template, Response,request
import cv2
import pymongo
import pandas as pd
import face_recognition
import numpy as np
import dlib
from fer import FER
import smtplib
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import pickle
import datetime
#Initialize the Flask app
app = Flask(__name__)

camera = cv2.VideoCapture(0)
client = pymongo.MongoClient("mongodb+srv://het:hetmongopassword@cluster0.hdmng.mongodb.net/myFirstDatabase?retryWrites=true&w=majority")
security_db = client.get_database('security_system')
protoPath = r'static\face_detector\deploy.prototxt'
modelPath = r'static\face_detector\res10_300x300_ssd_iter_140000.caffemodel'
model = load_model(r'static\liveness.model')
le = pickle.loads(open(r'static\le.pickle', "rb").read())
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
first_run = True
initial_time = datetime.datetime.utcnow()
s = smtplib.SMTP('smtp.gmail.com', 587)
s.starttls()
s.login("hetmehta61@gmail.com", "ysjqpwfsffrjonum")
check_live = []
check_emotion = []
detector = FER()
security_db.logs.insert_one({"time":datetime.datetime.now(),"log":"Security system turned on"})

def most_frequent(List):
    return max(set(List), key = List.count)

def detect_emotions(frame):
    global check_emotion
    global first_run
    global initial_time

    emotions = detector.detect_emotions(frame) # Detect emotions
    for i in emotions:
        emotion = i['emotions']
        dom_emotion = max(zip(emotion.values(), emotion.keys()))[1]
        check_emotion.append(dom_emotion)
        if len(check_emotion) == 10:
            res = most_frequent(check_emotion)
            print(res)
            if(res == 'angry' or res == 'fear'): # If extreme emotions alert user
                    time_now = datetime.datetime.utcnow()
                    if(((time_now - initial_time).total_seconds() > 300) or first_run):
                        message = 'Subject: {}\n\n{}'.format('SECURITY ALERT FROM HOUSE', "Someone with extreme emotions detected")    
                        s.sendmail("hetmehta61@gmail.com", "hetmehta61@gmail.com", message)                                                
                        security_db.logs.insert_one({"time":datetime.datetime.now(),"log":"Someone with extreme emotions detected"})
                        initial_time = time_now
                        first_run = False
            check_emotion = []

def detect_liveness(frame):
    global check_live
    global first_run
    global initial_time
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)

            face = frame[startY:endY, startX:endX]
            face = cv2.resize(face, (32, 32))
            face = face.astype("float") / 255.0
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0)
            preds = model.predict(face)[0]
            j = np.argmax(preds)
            label = le.classes_[j]
            check_live.append(label)
            if len(check_live) == 10:
                res = most_frequent(check_live)
                print(res)
                if res == 'fake':
                    time_now = datetime.datetime.utcnow()
                    if(((time_now - initial_time).total_seconds() > 300) or first_run):
                        message = 'Subject: {}\n\n{}'.format('SECURITY ALERT FROM HOUSE', "Someone unauthorized is trying to enter the premises")
                        s.sendmail("hetmehta61@gmail.com", "hetmehta61@gmail.com", message)
                        security_db.logs.insert_one({"time":datetime.datetime.now(),"log":"Someone unauthorized is trying to enter the premises"})                                                 
                        initial_time = time_now
                        first_run = False
                check_live = []   

def gen_frames():    
    global known_face_encodings 
    global known_face_names 
    global face_locations 
    global face_encodings 
    global face_names 
    global process_this_frame 
    global initial_time 

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)            
            rgb_small_frame = small_frame[:, :, ::-1]
            if process_this_frame:
                # Find all the faces and face encodings in the current frame of video
                detect_emotions(frame)
                detect_liveness(frame)
                
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                face_names = []
                for face_encoding in face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Unknown"

                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

                    face_names.append(name)
                if 'Unknown' in face_names: # If an unknown person comes, it will alert the user
                    time_now = datetime.datetime.utcnow()
                    if(((time_now - initial_time).total_seconds() > 300) or first_run):
                        message = 'Subject: {}\n\n{}'.format('SECURITY ALERT FROM HOUSE', "Unknown individual at your premises")
                        s.sendmail("hetmehta61@gmail.com", "hetmehta61@gmail.com", message)
                        print('Email Sent')
                        security_db.logs.insert_one({"time":datetime.datetime.now(),"log":"Unknown individual at your premises"})
                        initial_time = time_now
                        first_run = False
            process_this_frame = not process_this_frame
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
@app.route('/login',methods=['GET','POST'])
def login():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        username = request.form['username']
        password = request.form['password']
        user = security_db.login_details.find_one({
            'login_id': username,
            'password': password        
        })

        if user:            
            return render_template('about.html',title ='About Us')                                        
        else:
            msg = 'Invalid username or password'
            return render_template('login.html',msg = msg)                                        
    else:
        return render_template('login.html', msg = msg)


@app.route('/about')
def about():
    return render_template('about.html',title='About Us')

@app.route('/live')
def live():
    het_image = face_recognition.load_image_file("static\Het_pic.jpg")
    het_face_encoding = face_recognition.face_encodings(het_image)[0]
    global known_face_encodings
    global known_face_names
    global face_locations 
    global face_encodings 
    global face_names
    global process_this_frame

    known_face_encodings = [
       het_face_encoding
    ]
    known_face_names = [
        "Het Mehta"
    ]
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    return render_template('live.html')

@app.route('/notif')
def notif():
    recs = security_db.logs.find({}).sort('time',pymongo.DESCENDING)
    return render_template('notifications.html', recs = recs)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)