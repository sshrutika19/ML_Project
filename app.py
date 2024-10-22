import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from flask import Flask, request, jsonify, render_template,send_from_directory

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('pneumonia_Model.h5')


# Function to preprocess uploaded image
def preprocess_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/templates/<path:filename>')
def serve_static(filename):
    return send_from_directory('templates', filename)


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})

    # Get uploaded image file
    image = request.files['image']

    try:
        # Save the uploaded image to a temporary file
        image_path = "temp.jpg"
        image.save(image_path)

        # Preprocess the image
        processed_image = preprocess_image(image_path)

        # Debugging: Check processed image shape
        print("Processed image shape:", processed_image.shape)

        # Make prediction using the loaded model
        predictions = model.predict(processed_image)
        print("Predictions:", predictions)

        predicted_class = np.argmax(predictions[0])
        class_names = ['Normal', 'Pneumonia']
        predicted_label = class_names[predicted_class]
        confidence = float(predictions[0][predicted_class])

        # Delete temporary image file
        os.remove(image_path)

        # Return prediction result
        response = {
            'predicted_class': predicted_label,
            'confidence': confidence
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
