from arch_elements.model_evaluate import TensorflowModelPredictor

from flask import Flask, jsonify, request, url_for, render_template, flash, redirect
import tensorflow as tf
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import uuid
import time
import os

"""
Run using
```
FLASK_RUN_PORT=8080 FLASK_APP=arch_elements/model_deployment/app.py flask run --host=0.0.0.0
```
"""

app = Flask(__name__)
app.config['SECRET_KEY'] = str(uuid.uuid4())
app.config["MODEL"] = TensorflowModelPredictor('mobilenet')
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__),"feedbacks")

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    

@app.route('/')
def load_index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def load_prediction():
    if 'file' not in request.files:
        flash('No file part', 'error')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading', 'error')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        image = file.read()
        encoded_image = base64.b64encode(image).decode('utf-8')
        prediction = get_prediction(encoded_image)[0]
        other_classes = [ c for _, c in app.config["MODEL"].classes.items() if c != prediction ]

        flash('Predicted: ' + str(prediction), 'prediction')
        return render_template('index.html', image=encoded_image, predicted=prediction, other_categories=other_classes)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif', 'error')
        return redirect(request.url)

@app.route('/feedback', methods=['POST'])
def feedback():
    app.logger.error("Feedback value " + str(request.form.get("prediction_ok")))
    app.logger.error("Feedback value " + str(request.form.get("prediction_ko")))
    app.logger.error("returning you to " + str(app.config['UPLOAD_FOLDER']))
    f = request.form

    good_class = f.get("prediction_ok") if f.get("prediction_ok") else f.get("prediction_ko")
    image_array = decode_image(request.form.get("image"))
    name = good_class + '_' + str(time.time()) + '.png'
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], name)
    tf.keras.preprocessing.image.array_to_img(image_array).save(file_path)
    flash('Thank you for sending your feedback!', 'info')
    return redirect('/')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Provides endpoint for image classification prediction.

    HTTP request parameters
    -----------------------
    data : json
        Json POST containing the image in base64
    model : json


    HTTP response
    -------------
    prediction : json
        Predicted label by the model in the format:
        { prediction: label }

    Example request
    ---------------
    ```
    curl -X POST -H "Content-Type: application/json" \\
    --data "{\"data\":\"$(base64 -w0 image.jpg)\"}" \\
    http://127.0.0.1:8080/predict
    ```
    """
    data = request.get_json(silent=True)
    try:
        prediction = get_prediction(data['data'])
        app.logger.info("Predicted label s" + str(prediction))
        res = {
            'prediction': prediction[0].tolist(),
            }
        return jsonify(res)
    except KeyError as e:
        msg = 'Problem with the provided json post: {0}'.format(e)
        app.logger.error(msg)
        resp = jsonify({'error': msg})
        resp.status_code = 400
        return resp
    
def get_prediction(encoded_image:str) -> str:
    img = prepare_image(encoded_image)
    return app.config["MODEL"].predict(img)

def prepare_image(encoded_image:str) -> np.array:
    image = rescale_image(decode_image(encoded_image))
    return np.array([image])

def decode_image(encoded_image:str):
    img_b = BytesIO(base64.b64decode(encoded_image))
    return tf.keras.preprocessing.image.img_to_array(Image.open(img_b))

def rescale_image(image):
    return tf.keras.preprocessing.image.smart_resize(image, (128,128))

# local development ($ python star-rest.py)
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Start REST endpoint to predict the architectural elements of an image')
    parser.add_argument('--modeldir', metavar='path', required=True,
                        help='the path to the saved model')

    args = parser.parse_args()
    app.config["MODEL"] = TensorflowModelPredictor(args.modeldir)

    app.run(host="0.0.0.0", port=8080)

