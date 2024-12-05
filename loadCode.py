# predict_similarity.py

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Image parameters
img_width, img_height = 224, 224

# Function to preprocess a single image
def preprocess_image(file_path):
    img = load_img(file_path, target_size=(img_width, img_height))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img / 255.0

# Load the saved model
model = load_model('animal_classifier_model.h5')

# Function to predict class label
def predict_class(image_path):
    image = preprocess_image(image_path)
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)  # Get the index of the highest probability
    return predicted_class

# Compare two images
def check_similarity(img_path1, img_path2):
    class1 = predict_class(img_path1)
    class2 = predict_class(img_path2)

    if class1 == class2:
        print(f"The images contain the same object class: {class1}")
    else:
        print(f"The images contain different object classes: {class1} vs {class2}")

# Example usage
img1_path = 'path_to_image1.jpg'
img2_path = 'path_to_image2.jpg'
check_similarity(img1_path, img2_path)
