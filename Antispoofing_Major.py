                                            
                                            
                                       							 #---- DataCollection.py--#
from cvzone.FaceDetectionModule import FaceDetector
import cv2
import cvzone
from time import time
###################
classID = 0 #0 is fake and 1 is real
outputFolderPath = r'Dataset/DataCollect'
confidence = 0.8
save = True
blurThreshold = 30


debug = False
offsetPercentageW = 10
offsetPercentageH = 20
camWidth, camHeight = 640, 480
floatingPoint = 6
###################
cap = cv2.VideoCapture(0)
cap.set(3, camWidth)
cap.set(4, camHeight)
detector = FaceDetector()
while True:
        success, img = cap.read()
        imgOut = img.copy()
        img,bboxs = detector.findFaces(img, draw=False)

        listBlur = []
        listInfo = []

        # Check if any face is detected
        if bboxs:
            # Loop through each bounding box
            for bbox in bboxs:
                # bbox contains 'id', 'bbox', 'score', 'center'
                x,y,w,h = bbox["bbox"]
                score = bbox["score"][0]

                #print(x,y,w,h)
                #--------- Check----#

                if score>confidence:

                    #---Adding Offset----#
                    offsetW = (offsetPercentageW/100) * w
                    x = int(x- offsetW)
                    w=int(w+offsetW *2)

                    offsetH = (offsetPercentageH / 100) * h
                    y = int(y - offsetH*3)
                    h = int(h + offsetH *3.5)

                    #----- To avoid O----#
                    if x<0: x=0
                    if y<0: y=0
                    if w<0: w=0
                    if h<0: h=0

                    #----- Blurriness----#
                    imgFace = img[y:y+h, x:x+w]
                    cv2.imshow("Face", imgFace)
                    blurValue = int(cv2.Laplacian(imgFace, cv2.CV_64F).var())
                    if blurValue>blurThreshold:
                        listBlur.append(True)
                    else:
                        listBlur.append(False)

                    # -------------- Normalize Values--------#
                    ih,iw,_ = img.shape
                    xc,yc = x+w/2,y+h/2
                    xcn,ycn = round(xc /iw, floatingPoint),round(yc /ih, floatingPoint)
                    wn,hn = round(w/iw, floatingPoint),round(h/ih, floatingPoint)
                    #print(xcn,ycn,wn,hn)

                    #------ To avoid values above 1 ----#

                    if xcn>1: xcn=1
                    if ycn>1: ycn=1
                    if wn>1: wn=1
                    if hn>1: hn=1

                    listInfo.append(f"{classID} {xcn} {ycn} {wn} {hn}\n")



                    #-------Drawing----#
                    cv2.rectangle(imgOut,(x,y),(x+w,y+h),(0,255,0),3)
                    cvzone.putTextRect(imgOut,f'Score:{int(score*100)}% Blur: {blurValue}',(x,y-20),
                                       scale=2, thickness=3)
                    if debug:
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                        cvzone.putTextRect(img, f'Score:{int(score * 100)}% Blur: {blurValue}', (x, y - 20),
                                           scale=2, thickness=3)

                if save:
                    if all(listBlur) and listBlur != []:
                        timeNow = time()
                        timeNow = str(timeNow).split('.')
                        timeNow = timeNow[0]+timeNow[1]
                        print(timeNow)
                        cv2.imwrite(f"{outputFolderPath}/{timeNow}.jpg", img)
                        #----- Save Label File-------#
                        for info in listInfo:
                            f=open(f"{outputFolderPath}/{timeNow}.txt","a")
                            f.write(info)
                            f.close()



        # Display the image in a window named 'Image'
        cv2.imshow("Image", imgOut)
        # Wait for 1 millisecond, and keep the window open
        cv2.waitKey(1)












                                        					#-----FaceDetectorTest-----#

from cvzone.FaceDetectionModule import FaceDetector
import cv2
import cvzone
cap = cv2.VideoCapture(0)
detector = FaceDetector(minDetectionCon=0.5, modelSelection=0)
while True:
        # Read the current frame from the webcam
        # success: Boolean, whether the frame was successfully grabbed
        # img: the captured frame
        success, img = cap.read()

        # Detect faces in the image
        # img: Updated image
        # bboxs: List of bounding boxes around detected faces
        img, bboxs = detector.findFaces(img, draw=False)

        # Check if any face is detected
        if bboxs:
            # Loop through each bounding box
            for bbox in bboxs:
                # bbox contains 'id', 'bbox', 'score', 'center'

                # ---- Get Data  ---- #
                center = bbox["center"]
                x, y, w, h = bbox['bbox']
                score = int(bbox['score'][0] * 100)

                # ---- Draw Data  ---- #
                cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)
                cvzone.putTextRect(img, f'{score}%', (x, y - 10))
                cvzone.cornerRect(img, (x, y, w, h))

        # Display the image in a window named 'Image'
        cv2.imshow("Image", img)
        # Wait for 1 millisecond, and keep the window open
        cv2.waitKey(1)










                        							#---SplitData----#
import os
import random
import shutil
from itertools import islice

outputFolderPath = "Dataset/SplitData"
inputFolderPath ="Dataset/all"
splitRatio = {"train" : 0.7, "val" : 0.2,  "test" : 0.1}
classes = ["Real","Fake"]
try:
    shutil.rmtree(outputFolderPath)
    #print("Folder removed")
except OSError as e:
    os.makedirs(outputFolderPath)

# -- Create Directories---#

os.makedirs(f"{outputFolderPath}/train/images",exist_ok=True)
os.makedirs(f"{outputFolderPath}/train/labels",exist_ok=True)
os.makedirs(f"{outputFolderPath}/val/images",exist_ok=True)
os.makedirs(f"{outputFolderPath}/val/labels",exist_ok=True)
os.makedirs(f"{outputFolderPath}/test/images",exist_ok=True)
os.makedirs(f"{outputFolderPath}/test/labels",exist_ok=True)

listNames= os.listdir(inputFolderPath)
#print(listNames)
#print(len(listNames))

uniqueNames = []
for name in listNames:
    uniqueNames.append(name.split(".")[0])
uniqueNames = list(set(uniqueNames))

#print(set(uniqueNames))
#print(len(uniqueNames))

#-------- Shuffle----#
random.shuffle(uniqueNames)
#print(uniqueNames)

#-------Find the no of Images in a Folder---#
lenData = len(uniqueNames)
print(f'Totl images: {lenData}')
lenTrain = int(lenData*splitRatio["train"])
lenVal = int(lenData*splitRatio["val"])
lenTest = int(lenData*splitRatio["test"])
#print(f'Total images:{lenData} \n split : {lenTrain}  {lenVal}  {lenTest}')

# ----- Put Remaining images in training----#
if lenTrain != lenTrain+lenVal+lenTest:
   remaining = lenData-(lenTrain + lenVal + lenTest)
   lenTrain += remaining
#print(f'Total images:{lenData} \n split : {lenTrain}  {lenVal}  {lenTest}')
#------- Split the List -------#
lengthToSplit = [lenTrain, lenVal, lenTest]
Input = iter(uniqueNames)
Output = [list(islice(Input, elem)) for elem in lengthToSplit]  # Convert islice to list
#print(Output)
# Print dataset split sizes
#print(f'Total images: {lenData} \nSplit: {len(Output[0])} {len(Output[1])} {len(Output[2])}')


sequence = ['train', 'val', 'test']
for i, out in enumerate(Output):
    for filename in out:

        shutil.copy(f'{inputFolderPath}/{filename}.jpg', f'{outputFolderPath}/{sequence[i]}/images/{filename}.jpg')
        shutil.copy(f'{inputFolderPath}/{filename}.txt', f'{outputFolderPath}/{sequence[i]}/labels/{filename}.txt')

    #----------Yaml File created
dataYaml = f'path: ../Data\n\
train: ..train/images\n\
val: ..val/images\n\
test: ..test/images\n\
\n\
nc:{len(classes)}\n\
names: {classes}'

f = open(f"{outputFolderPath}/data.yaml", "a")
f.write(dataYaml)
f.close()











										#------- TestFileTest---#

f=open("test.txt","a")
f.write("This ia a New line\n")
f.close()














										#---------YoloTest----#	

from ultralytics import YOLO
import cv2
import cvzone
import math
import time

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Width
cap.set(4, 720)  # Height

# Load YOLO model (Ensure the correct path)
model = YOLO("../models/yolov8n.pt")  # Use correct YOLO model

# Define class names (Adjust as per model classes)
classNames = ["Real", "Fake"]

prev_frame_time = 0

while True:
    new_frame_time = time.time()  # Time for FPS calculation

    success, img = cap.read()
    if not success:
        break  # Stop if frame capture fails

    results = model(img, stream=True, verbose=False)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to int
            w, h = x2 - x1, y2 - y1
            conf = round(float(box.conf[0]), 2)  # Round confidence to 2 decimal places
            cls = int(box.cls[0])  # Get class index

            # Prevent IndexError (handle unknown classes)
            if cls < len(classNames):
                label = f'{classNames[cls]} {conf}'
            else:
                label = f'Class {cls} {conf}'  # Display class ID if unknown

            # Draw bounding box and label
            cvzone.cornerRect(img, (x1, y1, w, h), colorR=(0, 255, 0))
            cvzone.putTextRect(img, label, (max(0, x1), max(35, y1)), scale=1, thickness=2)

    # Calculate FPS
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show Image
    cv2.imshow("Image", img)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()







										#----------- Main-------#

import cvzone
from ultralytics import YOLO
import cv2
import cvzone
import math
import time

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Width
cap.set(4, 720)  # Height

# Load YOLO model (Ensure the correct path)
model = YOLO("../models/best_300.pt")  # Use correct YOLO model

# Define class names (Adjust as per model classes)
classNames = ["Fake", "Real"]

prev_frame_time = 0

while True:
    new_frame_time = time.time()  # Time for FPS calculation

    success, img = cap.read()
    if not success:
        break  # Stop if frame capture fails

    results = model(img, stream=True, verbose=False)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to int
            w, h = x2 - x1, y2 - y1
            conf = round(float(box.conf[0]), 2)  # Round confidence to 2 decimal places
            cls = int(box.cls[0])  # Get class index

            # Prevent IndexError (handle unknown classes)
            if cls < len(classNames):
                label = f'{classNames[cls]} {conf}'
            else:
                label = f'Class {cls} {conf}'  # Display class ID if unknown

            # Draw bounding box and label
            cvzone.cornerRect(img, (x1, y1, w, h), colorR=(0, 255, 0))
            cvzone.putTextRect(img, label, (max(0, x1), max(35, y1)), scale=1, thickness=2)

    # Calculate FPS
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show Image
    cv2.imshow("Image", img)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()








						#---- train----#
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
def main():

    model.train(data ='Testing Scripts/Dataset/SplitData/dataOffline.yaml',epochs =300)

if __name__ == "__main__":
    main()



						#-------------Streamlit--------------#
import streamlit as st
st.set_page_config(page_title="Face Login System", layout="centered")  # First Streamlit call

import cv2
from ultralytics import YOLO
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time

# Load YOLO model
yolo_model = YOLO("models/best_280.pt")
classNames = ["Fake", "Real"]

# Dataset of real faces
KNOWN_FACE_DIR = "Testing Scripts/Dataset/Real"

# Feature extraction (simple flattened vector)
def extract_features(image):
    image = cv2.resize(image, (100, 100))
    return image.flatten() / 255.0

# Load known real faces
@st.cache_data
def load_known_faces():
    faces = {}
    for file in os.listdir(KNOWN_FACE_DIR):
        if file.lower().endswith((".jpg")):
            img = cv2.imread(os.path.join(KNOWN_FACE_DIR, file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            name = os.path.splitext(file)[0]
            faces[name] = extract_features(img)
    return faces

known_faces = load_known_faces()

# Match face
def match_face(embedding, known_faces, threshold=0.8):
    for name, db_embedding in known_faces.items():
        similarity = cosine_similarity([embedding], [db_embedding])[0][0]
        if similarity > threshold:
            return name, similarity
    return None, None

# Streamlit UI
st.title("🔐  Face Login System")

if st.button("📸 Capture & Authenticate"):
    cap = cv2.VideoCapture(0)
    time.sleep(2)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        st.error("❌ Failed to capture image from webcam.")
    else:
        detected_face_crop = None
        label = None

        results = yolo_model(frame, stream=False)
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                if conf > 0.8:
                    label = classNames[cls]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detected_face_crop = frame[y1:y2, x1:x2]

                    color = (0, 255, 0) if label == "Real" else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label} {int(conf * 100)}%", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    break  # Use first detection only

        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Detection Result", use_container_width=True)

        if label == "Real" and detected_face_crop is not None:
            captured_embedding = extract_features(detected_face_crop)
            user, score = match_face(captured_embedding, known_faces)

            if user:
                st.success(f"✅ Welcome {user}! (Match: {score:.2f})")
                st.subheader("🏠 Home Page")
                st.markdown("You are now logged in successfully!")
            else:
                st.error("❌ Face is real, but not found in the dataset.")
        elif label == "Fake":
            st.error("🚫 Fake face detected. Access denied.")
        else:
            st.warning("⚠️ No valid face detected. Try again.")












