from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
from google.cloud import storage
import uuid
from google.oauth2 import service_account
from PIL import Image
import io
import base64

# Path to the service account key JSON file
service_account_key_path = 'serviceaccountkey.json'

# Create credentials using the service account key file
credentials = service_account.Credentials.from_service_account_file(
    service_account_key_path,
    scopes=['https://www.googleapis.com/auth/cloud-platform']
)

# Initialize Google Cloud Storage client with the credentials
storage_client = storage.Client(credentials=credentials)
bucket_name = 'ecoease'
bucket = storage_client.bucket(bucket_name)

# Load the model
model = load_model('model_trash.h5')

# Define class labels
class_labels = {0: 'glass', 1: 'paper', 2: 'cardboard', 3: 'plastic', 4: 'metal', 5: 'trash'}

# Initialize Flask app
app = Flask(__name__)

def preprocess_image(image):
    # Normalize pixel values to the range [0, 1]
    image = image / 255.0

    # Perform any other preprocessing steps such as resizing, cropping, etc.
    # ...

    return image

@app.route('/', methods=['GET'])
def success():
    return "Success! :)"

@app.route('/classify', methods=['POST'])
def classify_object():
    # Generate a unique name for the image
    unique_filename = str(uuid.uuid4()) + '.jpg'

    # Get the image file from the request
    image_file = request.files['image']

    # Save the file to a temporary location
    temp_path = 'temp_image.jpg'
    image_file.save(temp_path)

    # Upload the image to Google Cloud Storage
    blob = bucket.blob('images_ML/' + unique_filename)
    blob.upload_from_filename(temp_path)

    # Get the public URL of the uploaded image
    image_url = blob.public_url

    # Load and preprocess the image
    image = tf.keras.preprocessing.image.load_img(temp_path, target_size=(224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = preprocess_image(image)  # Preprocess the image as per your model's requirements

    # Add a batch dimension to the image
    image = tf.expand_dims(image, axis=0)

    # Perform inference
    predictions = model.predict(image)

    # Process the predictions
    predicted_classes = tf.argmax(predictions, axis=1)
    predicted_label = class_labels[predicted_classes[0].numpy()]

    # Convert the image to PIL Image
    image_pil = tf.keras.preprocessing.image.array_to_img(image[0])

    # Create a buffer to save the image
    image_buffer = io.BytesIO()

    # Save the PIL image to the buffer
    image_pil.save(image_buffer, format='JPEG')

    # Get the hexadecimal representation of the image buffer
    image_hex = image_buffer.getvalue().hex()

    # Delete the temporary file
    os.remove(temp_path)

    # Return the predicted object label and image buffer as the API response
    return jsonify({'predicted_label': predicted_label, 'image_hex': image_hex})

if __name__ == '__main__':
    app.run()
