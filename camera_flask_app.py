from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import datetime, time
import os, sys
import numpy as np
from threading import Thread
import glob
import face_recognition



global capture,rec_frame, grey, switch, neg, face, rec, out , flag, user
capture=0
grey=0
neg=0
face=0
switch=1
rec=0
flag = 0
user = 'unkwn'
#make shots directory to save pics
try:
    os.mkdir('./shots')
except OSError as error:
    pass

#Load pretrained face detection model    
#net = cv2.dnn.readNetFromCaffe('./saved_model/deploy.prototxt.txt', './saved_model/res10_300x300_ssd_iter_140000.caffemodel')

#instatiate flask app  
app = Flask(__name__, template_folder='./templates')


camera = cv2.VideoCapture(0)

def record(out):
    global rec_frame
    while(rec):
        time.sleep(0.05)
        out.write(rec_frame)

def detect_face(frame):
    """global net
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))   
    net.setInput(blob)
    detections = net.forward()
    confidence = detections[0, 0, 0, 2]

    if confidence < 0.5:            
            return frame           

    box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    try:
        frame=frame[startY:endY, startX:endX]
        (h, w) = frame.shape[:2]
        r = 480 / float(h)
        dim = ( int(w * r), 480)
        frame=cv2.resize(frame,dim)
    except Exception as e:
        pass
    """
    known_face_encodings = []
    known_face_names = []
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, 'known_people/')

    #make an array of all the saved jpg files' paths
    list_of_files = [f for f in glob.glob(path+'*.jpg')]
    #find number of known faces
    number_files = len(list_of_files)

    names = list_of_files.copy()

    for i in range(number_files):
        globals()['image_{}'.format(i)] = face_recognition.load_image_file(list_of_files[i])
        globals()['image_encoding_{}'.format(i)] = face_recognition.face_encodings(globals()['image_{}'.format(i)])[0]
        known_face_encodings.append(globals()['image_encoding_{}'.format(i)])

        # Create array of known names
        names[i] = names[i].replace("\\", "/")
        names[i] = names[i].replace("C:/Users/ASUS/project3/known_people/", "")
        names[i] = names[i].replace(".jpg", "")  
        known_face_names.append(names[i])
        #print(known_face_names)
    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    process_this_frame = True
    if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                face_names.append(name)
            #print(face_names)
            if (len(face_names)!=0):
                res = []
                [res.append(x) for x in face_names if x not in res]
                if len(res) ==1:
                        global user
                        user = res[0]
                        print(user)
    return frame

def gen_frames():  # generate frame by frame from camera
    global out, capture,rec_frame , usr, flag, user
    user = "unkwn"
    flag = 1
    while flag:
        success, frame = camera.read() 
        if success:
            if(flag):
                frame= detect_face(frame)                    
            # if(capture):
            #     capture=0
            #     now = datetime.datetime.now()
            #     p = os.path.sep.join(['shots', "shot_{}.png".format(str(now).replace(":",''))])
            #     cv2.imwrite(p, frame)
                
            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                # print(user)
                if (user == "unkwn"):
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                else:
                    now = datetime.datetime.now()
                    p = os.path.sep.join(['shots', user+"_{}.png".format(str(now).replace(":",''))])
                    cv2.imwrite(p, frame)
                    flag = 0
                    yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
                
        else:
            pass


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests',methods=['POST','GET'])
def tasks():
    global switch,camera
    if 1: #request.method == 'POST':
        global face
        face = 1
        if request.form.get('click') == '':
            global capture
            capture=1
        elif  request.form.get('face') == 'Login':
            
            # face=not face 
            if(face):
                # print(user)
                if user !='unkwn':
                    face=not face
                    if(switch==1):
                        switch = 0
                        camera.release()
                        cv2.destroyAllWindows()
                        return render_template('welcome.html',data = user)

                elif(user == "unkwn"):
                    redirect(url_for('tasks'))
                          
                 
    elif request.method=='GET':
        return render_template('index.html')
    return render_template('index.html')

if __name__ == '__main__':
    app.run()
