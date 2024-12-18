from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import io
from werkzeug.utils import secure_filename


model_cnn = tf.keras.models.load_model('./Model/model_vgg_2.h5')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
                return jsonify({"error": "No file part in the request"}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No file selected for uploading"}), 400

        # Convert FileStorage to io.BytesIO
        img_bytes = io.BytesIO(file.read())

        # Gunakan tf.keras.preprocessing untuk memproses gambar
        img = tf.keras.preprocessing.image.load_img(img_bytes, target_size=(180, 180))
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict using the model
        prediction = model_cnn.predict(img_array)
        predicted_class_index = np.argmax(prediction, axis=1)[0]

        
        class_names = ['acne', 'actinic keratosis', 'basal cell carcinoma', 'dermatofibroma', 'melanoma', 'nevus', 'normal', 'pigmented benign keratosis', 'seborrheic keratosis', 'squamous cell carcinoma', 'vascular lesion']
        
        predicted_class_name = class_names[predicted_class_index]
        
        print(predicted_class_name)
        print(predicted_class_index)

        return jsonify({"prediction": predicted_class_name})

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
