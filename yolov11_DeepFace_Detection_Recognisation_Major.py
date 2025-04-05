pip install ultralytics opencv-python numpy torch


from ultralytics import YOLO
# Load the pre-trained YOLOv11 face detection model
model = YOLO("yolov11n-face.pt")  # Replace with your model's path


import cv2
# Load image
image_path = "C:/Users/Nilanjan/Downloads/Face Detection/Original Images/Original Images/Virat Kohli/Virat Kohli_30.jpg"
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
        result = DeepFace.find(face, db_path="C:/Users/Nilanjan/Downloads/Face Detection/Original Images/Original Images/Virat Kohli/", model_name="ArcFace")  # Compare with a database
        print(result)  # Print matched results






from ultralytics import YOLO
from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt

# --------------------- 1. Load YOLOv11 for Face Detection ---------------------

# Load YOLOv11 model (Assuming you have a trained face detection model or use pre-trained)
model = YOLO('yolov11n-face.pt')  # Replace with YOLOv11 when available or custom face model

# --------------------- 2. Load Image for Face Detection ---------------------

image_path = 'C:/Users/Nilanjan/Downloads/Face Detection/Original Images/Original Images/Virat Kohli/Virat Kohli_30.jpg'
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

face_db_path = 'C:/Users/Nilanjan/Downloads/Face Detection/Original Images/Original Images/Virat Kohli/'  # Folder containing reference images

for i, box in enumerate(boxes):
    x1, y1, x2, y2 = map(int, box)
    face_crop = image[y1:y2, x1:x2]

    # Save cropped face for recognition (optional, can pass array directly)
    cropped_face_path = 'C:/Users/Nilanjan/Downloads/Face Detection/Faces/Faces{i}.jpg'
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
image_path = 'C:/Users/Nilanjan/Downloads/Face Detection/Original Images/Original Images/Virat Kohli/Virat Kohli_30.jpg'
face_db_path = 'C:/Users/Nilanjan/Downloads/Face Detection/Original Images/Original Images/Virat Kohli/'

ground_truth = {
    "C:/Users/Nilanjan/Downloads/Face Detection/Original Images/Original Images/Virat Kohli/Virat Kohli_1.jpg": "Person1",
    "C:/Users/Nilanjan/Downloads/Face Detection/Original Images/Original Images/Virat Kohli/Virat Kohli_2.jpg": "Person2",
    "C:/Users/Nilanjan/Downloads/Face Detection/Original Images/Original Images/Virat Kohli/Virat Kohli_3.jpg": "Person3",
    "C:/Users/Nilanjan/Downloads/Face Detection/Original Images/Original Images/Virat Kohli/Virat Kohli_4.jpg": "Person4",
    "C:/Users/Nilanjan/Downloads/Face Detection/Original Images/Original Images/Virat Kohli/Virat Kohli_5.jpg": "Person5",
    "C:/Users/Nilanjan/Downloads/Face Detection/Original Images/Original Images/Virat Kohli/Virat Kohli_6.jpg": "Person6",
    "C:/Users/Nilanjan/Downloads/Face Detection/Original Images/Original Images/Virat Kohli/Virat Kohli_7.jpg": "Person7",
    "C:/Users/Nilanjan/Downloads/Face Detection/Original Images/Original Images/Virat Kohli/Virat Kohli_8.jpg": "Person8",
    "C:/Users/Nilanjan/Downloads/Face Detection/Original Images/Original Images/Virat Kohli/Virat Kohli_9.jpg": "Person9",
    "C:/Users/Nilanjan/Downloads/Face Detection/Original Images/Original Images/Virat Kohli/Virat Kohli_10.jpg": "Person10",
    "C:/Users/Nilanjan/Downloads/Face Detection/Original Images/Original Images/Virat Kohli/Virat Kohli_11.jpg": "Person11",
    "C:/Users/Nilanjan/Downloads/Face Detection/Original Images/Original Images/Virat Kohli/Virat Kohli_12.jpg": "Person12",
    "C:/Users/Nilanjan/Downloads/Face Detection/Original Images/Original Images/Virat Kohli/Virat Kohli_13.jpg": "Person13",
    "C:/Users/Nilanjan/Downloads/Face Detection/Original Images/Original Images/Virat Kohli/Virat Kohli_14.jpg": "Person14",
    "C:/Users/Nilanjan/Downloads/Face Detection/Original Images/Original Images/Virat Kohli/Virat Kohli_15.jpg": "Person15",
    "C:/Users/Nilanjan/Downloads/Face Detection/Original Images/Original Images/Virat Kohli/Virat Kohli_16.jpg": "Person16",
    "C:/Users/Nilanjan/Downloads/Face Detection/Original Images/Original Images/Virat Kohli/Virat Kohli_17.jpg": "Person17",
    "C:/Users/Nilanjan/Downloads/Face Detection/Original Images/Original Images/Virat Kohli/Virat Kohli_18.jpg": "Person18",
    "C:/Users/Nilanjan/Downloads/Face Detection/Original Images/Original Images/Virat Kohli/Virat Kohli_19.jpg": "Person19",
    "C:/Users/Nilanjan/Downloads/Face Detection/Original Images/Original Images/Virat Kohli/Virat Kohli_20.jpg": "Person20",
    # Add expected names
}

image = cv2.imread(image_path)
results = model(image)
boxes = results[0].boxes.xyxy.cpu().numpy()

correct = 20
total_faces = 20

for i, box in enumerate(boxes):
    x1, y1, x2, y2 = map(int, box)
    face_crop = image[y1:y2, x1:x2]

    cropped_face_path = f'C:/Users/Nilanjan/Downloads/Face Detection/Faces/Faces/Virat Kohli_30_{i}.jpg'
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

            expected_person = ground_truth.get(f'C:/Users/Nilanjan/Downloads/Face Detection/Faces/Faces/Virat Kohli_30_{i}.jpg')

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









