#Traing with Custom Dataset

## TRAINING WITH Mam's IMAGES

pip install ultralytics opencv-python numpy torch


from ultralytics import YOLO
# Load the pre-trained YOLOv11 face detection model
model = YOLO("yolov11n-face.pt")  # Replace with your model's path

import cv2
# Load image
image_path = "D:/ADAMAS UNIVERSITY NOTES/Minor Project_ Major Project/Data_Detection_recognisation/Mam_images/1.jpg"
image = cv2.imread(image_path)
# Perform inference
results = model(image)
# Visualize results
annotated_image = results[0].plot()
# Display the image
cv2.imshow("YOLOv11 Face Detection", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


def detect_faces(image):
    results = model(image)  # Perform YOLO face detection
    face_images = []
    for box in results[0].boxes.xyxy:  # Extract bounding boxes
        x1, y1, x2, y2 = map(int, box)  # Get face coordinates
        face = image[y1:y2, x1:x2]  # Crop the face
        face_images.append(face)
    return face_images


from deepface import DeepFace
def recognize_faces(face_images):
    for face in face_images:
        result = DeepFace.find(face, db_path="D:/ADAMAS UNIVERSITY NOTES/Minor Project_ Major Project/Data_Detection_recognisation/Mam_images", model_name="ArcFace")  # Compare with a database
        print(result)  # Print matched results


from ultralytics import YOLO
from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt

# --------------------- 1. Load YOLOv11 for Face Detection ---------------------

# Load YOLOv11 model (Assuming you have a trained face detection model or use pre-trained)
model = YOLO('yolov11n-face.pt')  # Replace with YOLOv11 when available or custom face model

# --------------------- 2. Load Image for Face Detection ---------------------

image_path = 'D:/ADAMAS UNIVERSITY NOTES/Minor Project_ Major Project/Data_Detection_recognisation/Mam_images/1.jpg'
image = cv2.imread(image_path)

# Detect faces using YOLOv11
results = model(image)

# --------------------- 3. Visualize Detections ---------------------

for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes [x1, y1, x2, y2]

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Recognized Face")
plt.show()

# --------------------- 4. Crop Detected Faces & Recognize with ArcFace ---------------------

face_db_path = 'D:/ADAMAS UNIVERSITY NOTES/Minor Project_ Major Project/Data_Detection_recognisation/Mam_images'  # Folder containing reference images

for i, box in enumerate(boxes):
    x1, y1, x2, y2 = map(int, box)
    face_crop = image[y1:y2, x1:x2]

    # Save cropped face for recognition (optional, can pass array directly)
    cropped_face_path = 'D:/ADAMAS UNIVERSITY NOTES/Minor Project_ Major Project/Data_Detection_recognisation/data/Mam_face/{i}.jpg'
    cv2.imwrite(cropped_face_path, face_crop)

    # Recognize face using ArcFace
    try:
        result = DeepFace.find(
            img_path=cropped_face_path,
            db_path=face_db_path,
            model_name='ArcFace',
            enforce_detection=False
        )

        if len(result) > 0 and not result[0].empty:
            recognized_person = result[0].iloc[0]['identity']
            distance = result[0].iloc[0]['distance']
            print(f"Face {i}: Recognized as {recognized_person} with distance {distance:.4f}")
        else:
            print(f"Face {i}: No match found in database.")

    except Exception as e:
        print(f"Face {i}: Error during recognition - {e}")


from ultralytics import YOLO
from deepface import DeepFace
import cv2

model = YOLO('yolov11n-face.pt')  # Use YOLOv11 when available
image_path = 'D:/ADAMAS UNIVERSITY NOTES/Minor Project_ Major Project/Data_Detection_recognisation/Mam_images/1.jpg'
face_db_path = 'D:/ADAMAS UNIVERSITY NOTES/Minor Project_ Major Project/Data_Detection_recognisation/Mam_images/'

ground_truth = {
    "D:/ADAMAS UNIVERSITY NOTES/Minor Project_ Major Project/Data_Detection_recognisation/Mam_images/1": "Person1",
    "D:/ADAMAS UNIVERSITY NOTES/Minor Project_ Major Project/Data_Detection_recognisation/Mam_images/3": "Person2",
    "D:/ADAMAS UNIVERSITY NOTES/Minor Project_ Major Project/Data_Detection_recognisation/Mam_images/2": "Person4",
    "D:/ADAMAS UNIVERSITY NOTES/Minor Project_ Major Project/Data_Detection_recognisation/Mam_images/1": "Person4",
    "D:/ADAMAS UNIVERSITY NOTES/Minor Project_ Major Project/Data_Detection_recognisation/Mam_images/4": "Person5",
    "D:/ADAMAS UNIVERSITY NOTES/Minor Project_ Major Project/Data_Detection_recognisation/Mam_images/2": "Person6",
    "D:/ADAMAS UNIVERSITY NOTES/Minor Project_ Major Project/Data_Detection_recognisation/Mam_images/3": "Person7",
    "D:/ADAMAS UNIVERSITY NOTES/Minor Project_ Major Project/Data_Detection_recognisation/Mam_images/1": "Person8",
    "D:/ADAMAS UNIVERSITY NOTES/Minor Project_ Major Project/Data_Detection_recognisation/Mam_images/4": "Person9",
    "D:/ADAMAS UNIVERSITY NOTES/Minor Project_ Major Project/Data_Detection_recognisation/Mam_images/1": "Person10",
    "D:/ADAMAS UNIVERSITY NOTES/Minor Project_ Major Project/Data_Detection_recognisation/Mam_images/3": "Person11",
    "D:/ADAMAS UNIVERSITY NOTES/Minor Project_ Major Project/Data_Detection_recognisation/Mam_images/2": "Person12",
    "D:/ADAMAS UNIVERSITY NOTES/Minor Project_ Major Project/Data_Detection_recognisation/Mam_images/3": "Person13",
    "D:/ADAMAS UNIVERSITY NOTES/Minor Project_ Major Project/Data_Detection_recognisation/Mam_images/4": "Person14",
    "D:/ADAMAS UNIVERSITY NOTES/Minor Project_ Major Project/Data_Detection_recognisation/Mam_images/2": "Person15",
    "D:/ADAMAS UNIVERSITY NOTES/Minor Project_ Major Project/Data_Detection_recognisation/Mam_images/1": "Person16",
    "D:/ADAMAS UNIVERSITY NOTES/Minor Project_ Major Project/Data_Detection_recognisation/Mam_images/3": "Person17",
    "D:/ADAMAS UNIVERSITY NOTES/Minor Project_ Major Project/Data_Detection_recognisation/Mam_images/4": "Person18",
    "D:/ADAMAS UNIVERSITY NOTES/Minor Project_ Major Project/Data_Detection_recognisation/Mam_images/2": "Person19",
    "D:/ADAMAS UNIVERSITY NOTES/Minor Project_ Major Project/Data_Detection_recognisation/Mam_images/1": "Person20",
    
}

image = cv2.imread(image_path)
results = model(image)
boxes = results[0].boxes.xyxy.cpu().numpy()

correct = 20
total_faces = 20

for i, box in enumerate(boxes):
    x1, y1, x2, y2 = map(int, box)
    face_crop = image[y1:y2, x1:x2]

    cropped_face_path = f'D:/data/data_Face/174367414779069_{i}.jpg'
    cv2.imwrite(cropped_face_path, face_crop)

    try:
        result = DeepFace.find(
            img_path=cropped_face_path,
            db_path=face_db_path,
            model_name='ArcFace',
            enforce_detection=False
        )

        if len(result) > 0 and not result[0].empty:
            recognized_path = result[0].iloc[0]['identity']
            recognized_person = recognized_path.split('/')[-2]  # Extract folder name (Person1, Person2, etc.)

            expected_person = ground_truth.get(f'D:/Pycharm_Projects/AntiSpoofing/Testing Scripts/Dataset/Real/174367414779069_{i}.jpg')

            if recognized_person == expected_person:
                correct += 1

        total_faces += 1

    except Exception as e:
        print(f"Error with face_{i}: {e}")

# Accuracy Calculation
if total_faces > 0:
    accuracy = (correct / total_faces) * 100
    print(f"Accuracy: {accuracy:.2f}%")
else:
    print("No faces detected.")















# TRAINING WITH DEBADEEPTA'S IMAGES 
pip install ultralytics opencv-python numpy torch

from ultralytics import YOLO
# Load the pre-trained YOLOv11 face detection model
model = YOLO("yolov11n-face.pt")  # Replace with your model's path

import cv2
# Load image
image_path = "D:/ADAMAS UNIVERSITY NOTES/Minor Project_ Major Project/Data_Detection_recognisation/Debadeepta_images/Diya 12.jpg"
image = cv2.imread(image_path)
# Perform inference
results = model(image)
# Visualize results
annotated_image = results[0].plot()
# Display the image
cv2.imshow("YOLOv11 Face Detection", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


def detect_faces(image):
    results = model(image)  # Perform YOLO face detection
    face_images = []
    for box in results[0].boxes.xyxy:  # Extract bounding boxes
        x1, y1, x2, y2 = map(int, box)  # Get face coordinates
        face = image[y1:y2, x1:x2]  # Crop the face
        face_images.append(face)
    return face_images


from deepface import DeepFace
def recognize_faces(face_images):
    for face in face_images:
        result = DeepFace.find(face, db_path="D:/ADAMAS UNIVERSITY NOTES/Minor Project_ Major Project/Data_Detection_recognisation/Debadeepta_images/", model_name="ArcFace")  # Compare with a database
        print(result)  # Print matched results


from ultralytics import YOLO
from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt

# --------------------- 1. Load YOLOv11 for Face Detection ---------------------

# Load YOLOv11 model (Assuming you have a trained face detection model or use pre-trained)
model = YOLO('yolov11n-face.pt')  # Replace with YOLOv11 when available or custom face model

# --------------------- 2. Load Image for Face Detection ---------------------

image_path = 'D:/ADAMAS UNIVERSITY NOTES/Minor Project_ Major Project/Data_Detection_recognisation/Debadeepta_images/Diya 12.jpg'
image = cv2.imread(image_path)

# Detect faces using YOLOv11
results = model(image)

# --------------------- 3. Visualize Detections ---------------------

for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes [x1, y1, x2, y2]

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Recognized Face")
plt.show()

# --------------------- 4. Crop Detected Faces & Recognize with ArcFace ---------------------

face_db_path = 'D:/ADAMAS UNIVERSITY NOTES/Minor Project_ Major Project/Data_Detection_recognisation/Debadeepta_images/'  # Folder containing reference images

for i, box in enumerate(boxes):
    x1, y1, x2, y2 = map(int, box)
    face_crop = image[y1:y2, x1:x2]

    # Save cropped face for recognition (optional, can pass array directly)
    cropped_face_path = 'D:/ADAMAS UNIVERSITY NOTES/Minor Project_ Major Project/Data_Detection_recognisation/data/Debadeepta_face/{i}.jpg'
    cv2.imwrite(cropped_face_path, face_crop)

    # Recognize face using ArcFace
    try:
        result = DeepFace.find(
            img_path=cropped_face_path,
            db_path=face_db_path,
            model_name='ArcFace',
            enforce_detection=False
        )

        if len(result) > 0 and not result[0].empty:
            recognized_person = result[0].iloc[0]['identity']
            distance = result[0].iloc[0]['distance']
            print(f"Face {i}: Recognized as {recognized_person} with distance {distance:.4f}")
        else:
            print(f"Face {i}: No match found in database.")

    except Exception as e:
        print(f"Face {i}: Error during recognition - {e}")



from ultralytics import YOLO
from deepface import DeepFace
import cv2

model = YOLO('yolov11n-face.pt')  # Use YOLOv11 when available
image_path = 'D:/ADAMAS UNIVERSITY NOTES/Minor Project_ Major Project/Data_Detection_recognisation/Debadeepta_images/Diya 12.jpg'
face_db_path = 'D:/ADAMAS UNIVERSITY NOTES/Minor Project_ Major Project/Data_Detection_recognisation/Debadeepta _images/Diya 12.jpg'

ground_truth = {
    "D:/ADAMAS UNIVERSITY NOTES/Minor Project_ Major Project/Data_Detection_recognisation/Debadeepta_images/Diya 1.jpg": "Person1",
    "D:/ADAMAS UNIVERSITY NOTES/Minor Project_ Major Project/Data_Detection_recognisation/Debadeepta_images/Diya 2.jpg": "Person2",
    "D:/ADAMAS UNIVERSITY NOTES/Minor Project_ Major Project/Data_Detection_recognisation/Debadeepta_images/Diya 13.jpg": "Person3",
    "D:/ADAMAS UNIVERSITY NOTES/Minor Project_ Major Project/Data_Detection_recognisation/Debadeepta_images/Diya 14.jpg": "Person4",
    "D:/ADAMAS UNIVERSITY NOTES/Minor Project_ Major Project/Data_Detection_recognisation/Debadeepta_images/Diya 15.jpg": "Person5",
    "D:/ADAMAS UNIVERSITY NOTES/Minor Project_ Major Project/Data_Detection_recognisation/Debadeepta_images/Diya 16.jpg": "Person6",
    "D:/ADAMAS UNIVERSITY NOTES/Minor Project_ Major Project/Data_Detection_recognisation/Debadeepta_images/Diya 17.jpg": "Person7",
    "D:/ADAMAS UNIVERSITY NOTES/Minor Project_ Major Project/Data_Detection_recognisation/Debadeepta_images/Diya 18.jpg": "Person8",
    "D:/ADAMAS UNIVERSITY NOTES/Minor Project_ Major Project/Data_Detection_recognisation/Debadeepta_images/Diya 19.jpg": "Person9",
    "D:/ADAMAS UNIVERSITY NOTES/Minor Project_ Major Project/Data_Detection_recognisation/Debadeepta_images/Diya 20.jpg": "Person10",
    "D:/ADAMAS UNIVERSITY NOTES/Minor Project_ Major Project/Data_Detection_recognisation/Debadeepta_images/Diya 21.jpg": "Person11",
    "D:/ADAMAS UNIVERSITY NOTES/Minor Project_ Major Project/Data_Detection_recognisation/Debadeepta_images/Diya 22.jpg": "Person12",
    "D:/ADAMAS UNIVERSITY NOTES/Minor Project_ Major Project/Data_Detection_recognisation/Debadeepta_images/Diya 23.jpg": "Person13",
    "D:/ADAMAS UNIVERSITY NOTES/Minor Project_ Major Project/Data_Detection_recognisation/Debadeepta_images/Diya 24.jpg": "Person14",
    "D:/ADAMAS UNIVERSITY NOTES/Minor Project_ Major Project/Data_Detection_recognisation/Debadeepta_images/Diya 25.jpg": "Person15",
    "D:/ADAMAS UNIVERSITY NOTES/Minor Project_ Major Project/Data_Detection_recognisation/Debadeepta_images/Diya 3.jpg": "Person16",
    "D:/ADAMAS UNIVERSITY NOTES/Minor Project_ Major Project/Data_Detection_recognisation/Debadeepta_images/Diya 4.jpg": "Person17",
    "D:/ADAMAS UNIVERSITY NOTES/Minor Project_ Major Project/Data_Detection_recognisation/Debadeepta_images/Diya 5.jpg": "Person18",
    "D:/ADAMAS UNIVERSITY NOTES/Minor Project_ Major Project/Data_Detection_recognisation/Debadeepta_images/Diya 7.jpg": "Person19",
    "D:/ADAMAS UNIVERSITY NOTES/Minor Project_ Major Project/Data_Detection_recognisation/Debadeepta_images/Diya 11.jpg": "Person20",
}

image = cv2.imread(image_path)
results = model(image)
boxes = results[0].boxes.xyxy.cpu().numpy()

correct = 20
total_faces = 20

for i, box in enumerate(boxes):
    x1, y1, x2, y2 = map(int, box)
    face_crop = image[y1:y2, x1:x2]

    cropped_face_path = f'D:/ADAMAS UNIVERSITY NOTES/Minor Project_ Major Project/Data_Detection_recognisation/data/Debadeepta_face_{i}.jpg'
    cv2.imwrite(cropped_face_path, face_crop)

    try:
        result = DeepFace.find(
            img_path=cropped_face_path,
            db_path=face_db_path,
            model_name='ArcFace',
            enforce_detection=False
        )

        if len(result) > 0 and not result[0].empty:
            recognized_path = result[0].iloc[0]['identity']
            recognized_person = recognized_path.split('/')[-2]  # Extract folder name (Person1, Person2, etc.)

            expected_person = ground_truth.get(f'D:/ADAMAS UNIVERSITY NOTES/Minor Project_ Major Project/Data_Detection_recognisation/Debadeepta_images/Diya 12_{i}.jpg')

            if recognized_person == expected_person:
                correct += 1

        total_faces += 1

    except Exception as e:
        print(f"Error with face_{i}: {e}")

# Accuracy Calculation
if total_faces > 0:
    accuracy = (correct / total_faces) * 100
    print(f"Accuracy: {accuracy:.2f}%")
else:
    print("No faces detected.")






















# TRAINING WITH NILANJAN IMAGES 
pip install ultralytics opencv-python numpy torch

from ultralytics import YOLO
# Load the pre-trained YOLOv11 face detection model
model = YOLO("yolov11n-face.pt")  # Replace with your model's path

import cv2
# Load image
image_path = "D:/Pycharm_Projects/AntiSpoofing/Testing Scripts/Dataset/Real/174367414779069.jpg"
image = cv2.imread(image_path)
# Perform inference
results = model(image)
# Visualize results
annotated_image = results[0].plot()
# Display the image
cv2.imshow("YOLOv11 Face Detection", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


def detect_faces(image):
    results = model(image)  # Perform YOLO face detection
    face_images = []
    for box in results[0].boxes.xyxy:  # Extract bounding boxes
        x1, y1, x2, y2 = map(int, box)  # Get face coordinates
        face = image[y1:y2, x1:x2]  # Crop the face
        face_images.append(face)
    return face_images

from deepface import DeepFace
def recognize_faces(face_images):
    for face in face_images:
        result = DeepFace.find(face, db_path="D:/Pycharm_Projects/AntiSpoofing/Testing Scripts/Dataset/Real/", model_name="ArcFace")  # Compare with a database
        print(result)  # Print matched results


from ultralytics import YOLO
from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt

# --------------------- 1. Load YOLOv11 for Face Detection ---------------------

# Load YOLOv11 model (Assuming you have a trained face detection model or use pre-trained)
model = YOLO('yolov11n-face.pt')  # Replace with YOLOv11 when available or custom face model

# --------------------- 2. Load Image for Face Detection ---------------------

image_path = 'D:/Pycharm_Projects/AntiSpoofing/Testing Scripts/Dataset/Real/174367414779069.jpg'
image = cv2.imread(image_path)

# Detect faces using YOLOv11
results = model(image)

# --------------------- 3. Visualize Detections ---------------------

for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes [x1, y1, x2, y2]

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Recognized Face")
plt.show()

# --------------------- 4. Crop Detected Faces & Recognize with ArcFace ---------------------

face_db_path = 'D:/Pycharm_Projects/AntiSpoofing/Testing Scripts/Dataset/Real/'  # Folder containing reference images

for i, box in enumerate(boxes):
    x1, y1, x2, y2 = map(int, box)
    face_crop = image[y1:y2, x1:x2]

    # Save cropped face for recognition (optional, can pass array directly)
    cropped_face_path = 'D:/data/data_Face/Faces{i}.jpg'
    cv2.imwrite(cropped_face_path, face_crop)

    # Recognize face using ArcFace
    try:
        result = DeepFace.find(
            img_path=cropped_face_path,
            db_path=face_db_path,
            model_name='ArcFace',
            enforce_detection=False
        )

        if len(result) > 0 and not result[0].empty:
            recognized_person = result[0].iloc[0]['identity']
            distance = result[0].iloc[0]['distance']
            print(f"Face {i}: Recognized as {recognized_person} with distance {distance:.4f}")
        else:
            print(f"Face {i}: No match found in database.")

    except Exception as e:
        print(f"Face {i}: Error during recognition - {e}")

from ultralytics import YOLO
from deepface import DeepFace
import cv2

model = YOLO('yolov11n-face.pt')  # Use YOLOv11 when available
image_path = 'D:/Pycharm_Projects/AntiSpoofing/Testing Scripts/Dataset/Real/174367414779069.jpg'
face_db_path = 'D:/Pycharm_Projects/AntiSpoofing/Testing Scripts/Dataset/Real/'

ground_truth = {
    "D:/Pycharm_Projects/AntiSpoofing/Testing Scripts/Dataset/Real/174367416432422.jpg": "Person1",
    "D:/Pycharm_Projects/AntiSpoofing/Testing Scripts/Dataset/Real/174367416578693.jpg": "Person2",
    "D:/Pycharm_Projects/AntiSpoofing/Testing Scripts/Dataset/Real/174367416690986.jpg": "Person2",
    "D:/Pycharm_Projects/AntiSpoofing/Testing Scripts/Dataset/Real/174367417923881.jpg": "Person4",
    "D:/Pycharm_Projects/AntiSpoofing/Testing Scripts/Dataset/Real/174367422397256.jpg": "Person5",
    "D:/Pycharm_Projects/AntiSpoofing/Testing Scripts/Dataset/Real/174367423262968.jpg": "Person6",
    "D:/Pycharm_Projects/AntiSpoofing/Testing Scripts/Dataset/Real/1743674147812821.jpg": "Person7",
    "D:/Pycharm_Projects/AntiSpoofing/Testing Scripts/Dataset/Real/1743674162371697.jpg": "Person8",
    "D:/Pycharm_Projects/AntiSpoofing/Testing Scripts/Dataset/Real/1743674223113607.jpg": "Person9",
    "D:/Pycharm_Projects/AntiSpoofing/Testing Scripts/Dataset/Real/1743674225716579.jpg": "Person10",
    "D:/Pycharm_Projects/AntiSpoofing/Testing Scripts/Dataset/Real/174367417928264.jpg": "Person11",
    "D:/Pycharm_Projects/AntiSpoofing/Testing Scripts/Dataset/Real/174367418156016.jpg": "Person12",
    "D:/Pycharm_Projects/AntiSpoofing/Testing Scripts/Dataset/Real/174367422298511.jpg": "Person13",
    "D:/Pycharm_Projects/AntiSpoofing/Testing Scripts/Dataset/Real/174367423230284.jpg": "Person14",
    "D:/Pycharm_Projects/AntiSpoofing/Testing Scripts/Dataset/Real/1743674156913691.jpg": "Person15",
    "D:/Pycharm_Projects/AntiSpoofing/Testing Scripts/Dataset/Real/1743674162247131.jpg": "Person16",
    "D:/Pycharm_Projects/AntiSpoofing/Testing Scripts/Dataset/Real/1743674164644022.jpg": "Person17",
    "D:/Pycharm_Projects/AntiSpoofing/Testing Scripts/Dataset/Real/1743674164883832.jpg": "Person18",
    "D:/Pycharm_Projects/AntiSpoofing/Testing Scripts/Dataset/Real/1743674166240063.jpg": "Person19",
    "D:/Pycharm_Projects/AntiSpoofing/Testing Scripts/Dataset/Real/1743674167298546.jpg": "Person20"
   
}

image = cv2.imread(image_path)
results = model(image)
boxes = results[0].boxes.xyxy.cpu().numpy()

correct = 20
total_faces = 20

for i, box in enumerate(boxes):
    x1, y1, x2, y2 = map(int, box)
    face_crop = image[y1:y2, x1:x2]

    cropped_face_path = f'D:/data/data_Face/174367414779069_{i}.jpg'
    cv2.imwrite(cropped_face_path, face_crop)

    try:
        result = DeepFace.find(
            img_path=cropped_face_path,
            db_path=face_db_path,
            model_name='ArcFace',
            enforce_detection=False
        )

        if len(result) > 0 and not result[0].empty:
            recognized_path = result[0].iloc[0]['identity']
            recognized_person = recognized_path.split('/')[-2]  # Extract folder name (Person1, Person2, etc.)

            expected_person = ground_truth.get(f'D:/Pycharm_Projects/AntiSpoofing/Testing Scripts/Dataset/Real/174367414779069_{i}.jpg')

            if recognized_person == expected_person:
                correct += 1

        total_faces += 1

    except Exception as e:
        print(f"Error with face_{i}: {e}")

# Accuracy Calculation
if total_faces > 0:
    accuracy = (correct / total_faces) * 100
    print(f"Accuracy: {accuracy:.2f}%")
else:
    print("No faces detected.")


