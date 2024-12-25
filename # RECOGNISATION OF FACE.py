# RECOGNISATION OF A FACE




from scipy.spatial import distance

# Function to recognize a face
def recognize_face(image_path, known_face_embeddings, known_face_names, threshold=0.6):
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = face_detector(rgb_image)

    for face in faces:
        shape = shape_predictor(rgb_image, face)
        embedding = np.array(face_recognition_model.compute_face_descriptor(rgb_image, shape))

        # Initialize variables for matching
        name = "Vijay"
        min_distance = threshold

        # Compare with dataset embeddings
        for known_embedding, known_name in zip(known_face_embeddings, known_face_names):
            dist = distance.euclidean(embedding, known_embedding)
            if dist < min_distance:
                name = known_name
                min_distance = dist

        # Draw a rectangle and label on the image
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Display the result
    cv2.imshow("Face Recognition", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Path to the image to recognize
input_image_path = "C:/Users/Nilanjan/Downloads/Face Detection/Original Images/Original Images/Vijay Deverakonda/Vijay Deverakonda_50.jpg"

# Recognize the face
recognize_face(input_image_path, known_face_embeddings, known_face_names)






                
