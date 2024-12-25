# FACE DETECTOR






import dlib
import cv2

# Load Dlib's face detector
cnn_face_detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")

# Load an image
image_path = "C:/Users/Nilanjan/Downloads/Face Detection/Original Images/Original Images/Vijay Deverakonda/Vijay Deverakonda_50.jpg"
image = cv2.imread(image_path)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Detect faces using CNN
faces = cnn_face_detector(rgb_image, upsample_num_times=1)

# Draw rectangles around detected faces
for face in faces:
    x, y, w, h = (face.rect.left(), face.rect.top(), face.rect.width(), face.rect.height())
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Display the result
cv2.imshow("Face Detection with CNN", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
