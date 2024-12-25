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

