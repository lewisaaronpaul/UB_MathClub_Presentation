import os

#::: Import modules and packages :::
# Flask utils
from flask import Flask, redirect, url_for, request, render_template, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

from numpy import set_printoptions

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

import numpy as np
import matplotlib.pylab as plt

app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1

# Initial Settings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
set_printoptions(precision=4, suppress=True)

# Read in the model with MobileNet
modelMobileNet = tf.keras.models.load_model(('model/architecture_cnn_MobileNet_0.93.h5'),custom_objects={'KerasLayer':hub.KerasLayer})

print(modelMobileNet.summary())

# Model created with MobileNet
def classify_image(img_path, model, labels):
  dim = (224, 224)
  # Load image
  img = image.load_img(img_path, target_size = (224, 224))
  # Convert image to numpy array
  img = np.array(img)/255.0
  # Make prediction
  prediction = model.predict(img[np.newaxis,...])
  prediction_class = np.argmax(prediction[0], axis = -1)
  return labels[prediction_class].title()

image_folder = os.path.join('static', 'images')
app.config['UPLOAD_FLOADER'] = image_folder

@app.route('/')
def home():
	return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
  label_list = ['altar', 'apse', 'bell_tower', 'column', 'dome(inner)', 'dome(outer)', 'flying_buttress', 'gargoyle', 'stained_glass', 'vault']

  if request.method == "POST":
    img = request.files["picFile"]
    filename = secure_filename(img.filename)
    pic_path = os.path.join(app.config['UPLOAD_FLOADER'], filename)
    img.save(pic_path)
    # Predict with the MobileNet model
    the_prediction = classify_image(pic_path, modelMobileNet, label_list)
  else:
    print("Select an image")

  return render_template('predict.html', the_prediction = the_prediction, image_name = filename)

@app.route('/load_img/<filename>')
def load_img(filename):
  return send_from_directory(image_folder, filename)

if __name__ == '__main__':
	app.run(debug=False)

