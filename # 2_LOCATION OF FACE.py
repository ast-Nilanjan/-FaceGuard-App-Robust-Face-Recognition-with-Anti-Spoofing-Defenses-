# LOCATION OF FACE




# Import dlib after installation
import dlib

# Load a face detector
detector = dlib.get_frontal_face_detector()

# Load an image
image = dlib.load_rgb_image("C:/Users/Nilanjan/Downloads/Face Detection/Original Images/Original Images/Vijay Deverakonda/Vijay Deverakonda_50.jpg")

# Detect faces
faces = detector(image)

# Output the number of faces detected
print("Number of faces detected: ", len(faces))

# Loop over each detected face and print its position
for face in faces:
    print("Left: ", face.left())
    print("Top: ", face.top())
    print("Right: ", face.right())
    print("Bottom: ", face.bottom())
