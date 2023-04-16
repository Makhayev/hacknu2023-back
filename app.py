from flask import Flask
import numpy as np
import cv2
import argparse
from keras.models import load_model
import mediapipe as mp
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from io import BytesIO
from PIL import Image




app = Flask(__name__)

cors = CORS(app)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
cors = CORS(app, resources={r"/api/*": {"methods": ["GET", "POST", "PUT", "DELETE"]}})



mp_holistic = mp.solutions.holistic # Holistic model


holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

model = load_model('face.h5')
def extract_keypoints(results):
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    return np.concatenate([face])

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

people = ['021013650266', '030727651379', '000525501114', '000821550825']


@app.route('/foo', methods=['POST']) 
def foo():
    image_data = request.get_json()
    b64_str = image_data['video_frame'].split(',')[1]
    bytes_data = base64.b64decode(b64_str)

    # convert bytes to PIL Image object
    img = Image.open(BytesIO(bytes_data))

    # convert PIL Image to numpy array
    np_array = np.array(img)

    _, results = mediapipe_detection(np_array, holistic)
    keypoints = extract_keypoints(results)
    res = model.predict(np.expand_dims(keypoints, axis=0))
    print(people[np.argmax(res)])
    response = {'status': 'success', 'message': f'{people[np.argmax(res)]}'}
    

    return jsonify(response)


app.run(host='0.0.0.0', port=81)