import os
from flask_cors import CORS
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model
model = load_model(r'D:\Classification Project\Datasets\animal_classifier_model.h5')

# Define image dimensions and threshold
img_width, img_height = 224, 224
confidence_threshold = 0.75  # Set a threshold for model confidence

# Function to preprocess image
def preprocess_image(file_path):
    img = load_img(file_path, target_size=(img_width, img_height))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img / 255.0

# Function to predict the class of an image
def predict_class(image):
    prediction = model.predict(image)
    class_id = np.argmax(prediction)
    confidence = np.max(prediction)  # Get the highest confidence score
    return class_id, confidence

# Define /compare route to handle POST requests for image comparison
@app.route('/compare', methods=['POST', 'GET'])
def compare_images():
    if request.method == 'GET':
        return jsonify({'message': 'Use POST to upload images for comparison.'}), 405
    
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({'error': 'Please upload two images'}), 400

    image1 = request.files['image1']
    image2 = request.files['image2']

    # Create a temporary directory to save images
    os.makedirs('temp', exist_ok=True)
    path1 = os.path.join('temp', 'image1.jpg')
    path2 = os.path.join('temp', 'image2.jpg')
    
    # Save uploaded images to temporary paths
    image1.save(path1)
    image2.save(path2)

    # Preprocess and predict classes with confidence
    processed_image1 = preprocess_image(path1)
    processed_image2 = preprocess_image(path2)
    
    class1, confidence1 = predict_class(processed_image1)
    class2, confidence2 = predict_class(processed_image2)

    # Remove temporary images
    os.remove(path1)
    os.remove(path2)

    # Check confidence levels for both images
    if confidence1 < confidence_threshold or confidence2 < confidence_threshold:
        return jsonify({
            'result': 'the model is not trained on one or both of these images.'
        })

    # Determine similarity if both images are confidently recognized
    result = 'similar' if class1 == class2 else 'different'
    return jsonify({
        'result': result,
        'class1': int(class1),
        'confidence1': float(confidence1),
        'class2': int(class2),
        'confidence2': float(confidence2)
    })

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
