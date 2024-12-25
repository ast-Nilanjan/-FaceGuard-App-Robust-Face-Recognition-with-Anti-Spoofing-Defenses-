
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
