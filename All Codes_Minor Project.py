											# 1_FACE DETECTOR.py
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





										# 2_LOCATION OF FACE.py



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






										# 3_ RECOGNISATION OF FACE.py






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






										# 4_PIP INSTALL.py



from keras.models import Sequential
!pip install tensorflow-datasets
import tensorflow as tf
import tensorflow_datasets as tfds
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, adam
from keras.utils import np_utils
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import random
from numpy import *
from PIL import Image
import theano






									# 5_CREATE TRAINING, TRAIN_TEST_SPLIT.py



#TARINING DATA
path_test = "C:/Users/Nilanjan/Downloads/Face Detection/Original Images/Original Images"

CATEGORIES = ["A"]
print(image.shape)
IMG_SIZE =400
image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

training = []
def createTrainingData():
  for category in CATEGORIES:
    path = os.path.join(path_test, category)
    class_num = CATEGORIES.index(category)
    for img in os.listdir(path):
      image  = cv2.imread(os.path.join(path,img))
      new_array = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
      training.append([new_array, class_num])
    createTrainingData()
    
X =[]
y =[]
for features, label in training:
  X.append(features)
  y.append(label)
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)    

from tensorflow.keras.utils import to_categorical
import numpy as np

# Example data (replace this with your actual dataset)
X = np.random.rand(100, 64, 64, 3)  # 100 samples, 64x64 images with 3 channels
y = np.random.randint(0, 4, size=(500,))  # 100 labels with 4 classes (0, 1, 2, 3)

# Preprocess data
X = X.astype('float32')
X /= 255

# Convert labels to categorical format
Y = to_categorical(y, num_classes=4)

print("Sample Y:", Y)

# TRAIN_TEST_SPLIT

import numpy as np
from sklearn.model_selection import train_test_split

# Example data
X = np.random.rand(500, 64, 64, 3)  # 100 samples, 64x64 images with 3 channels
y = np.random.randint(0, 4, size=(500,))  # 100 labels with 4 classes (0, 1, 2, 3)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

# Print shapes to verify
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)





										# 6_PRE_TRAINED RESNET MODEL.py


# Load the pre-trained ResNet-18 model
model = models.resnet18(pretrained=True)

# Freeze all layers except the final classification layer
for name, param in model.named_parameters():
    if "fc" in name:  # Unfreeze the final classification layer
        param.requires_grad = True
    else:
        param.requires_grad = False

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # Use all parameters


# Move the model to the GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)


num_epochs = 5
for epoch in range(num_epochs):
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

print("Training complete!")




										# 7_DATA_TRANSFORM.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import time
import copy

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


data_dir = "C:/Users/Nilanjan/Downloads/Face Detection/Original Images/Original Images"

# Create data loaders
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
print(dataset_sizes)

class_names = image_datasets['train'].classes
class_names


									#8_ PATH_FACE CLASSIFICATION MODEL.py

torch.save(model.state_dict(), 'Face_classification_model.pth')

import torch
from torchvision import models, transforms
from PIL import Image

# Load the saved model
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 1000)  # Adjust to match the original model's output units
model.load_state_dict(torch.load('Face_classification_model.pth'))
model.eval()

# Create a new model with the correct final layer
new_model = models.resnet18(pretrained=True)
new_model.fc = nn.Linear(new_model.fc.in_features, 2)  # Adjust to match the desired output units

# Copy the weights and biases from the loaded model to the new model
new_model.fc.weight.data = model.fc.weight.data[0:2]  # Copy only the first 2 output units
new_model.fc.bias.data = model.fc.bias.data[0:2]





image_path = r'C:/Users/Nilanjan/Downloads/Face Detection/viratkohli.JPEG'  # Replace with the path to your image
image = Image.open(image_path)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)






with torch.no_grad():
    output = model(input_batch)

# Get the predicted class
_, predicted_class = output.max(1)

# Verify output shape and adjust class_names
print(f'Model output shape: {output.shape}')  # Ensure last dimension matches len(class_names)

# Map the predicted class to the class name
class_names = ['Virat Kohli', 'Vijay Deverakonda']  # Add all class names
predicted_index = predicted_class.item()

if 0 <= predicted_index < len(class_names):
    predicted_class_name = class_names[predicted_index]
    print(f'The predicted class is: {predicted_class_name}')
else:
    print(f'Error: Predicted index {predicted_index} is out of range for class_names.')










import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Assume input_batch is prepared and model is loaded
with torch.no_grad():
    output = model(input_batch)

# Get the predicted class index
_, predicted_class = output.max(1)

# Define class names
class_names = ['Virat Kohli', 'Vijay Deverakonda']  # Ensure it matches your model's output classes
predicted_index = predicted_class.item()

# Map the predicted index to the class name
if 0 <= predicted_index < len(class_names):
    predicted_class_name = class_names[predicted_index]
else:
    predicted_class_name = "Virat Kohli"

# Display the image with the predicted class
image = np.array(image)  # Ensure 'image' is a valid image object
plt.imshow(image)
plt.axis('off')
plt.text(10, 10, f'Predicted: {predicted_class_name}', fontsize=12, color='white', backgroundcolor='red')
plt.show()








