from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
from methods import process_and_crop_image, batch_infer_trocr, aqg


app = Flask(__name__)



@app.route('/process', methods=['POST'])
def process():
    try:
        # Get the Base64-encoded image from the request
        data = request.get_json()
        image_b64 = data['image']

        # Decode Base64 to binary and then to an OpenCV image
        image_binary = base64.b64decode(image_b64)
        image = cv2.imdecode(np.frombuffer(image_binary, np.uint8), cv2.IMREAD_GRAYSCALE)

        # Process the image
        cropped_images_array = process_and_crop_image(image)
        corrected_text = batch_infer_trocr(cropped_images_array)
        aqg_text = aqg(corrected_text)

        return jsonify({"text": aqg_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

app.run(host='0.0.0.0', port=6969)
