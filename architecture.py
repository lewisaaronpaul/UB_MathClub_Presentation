import os

#::: Import modules and packages :::
# Flask utils
from flask import Flask, redirect, url_for, request, render_template, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

from numpy import set_printoptions

import tensorflow as tf
import tensorflow_hub as hub
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# import pandas as pd
import numpy as np
# import matplotlib.pylab as plt
# import seaborn as sns

# import cv2

COUNT = 0
app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1

# Initial Settings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
set_printoptions(precision=4, suppress=True)

# Read in the model with MobileNet
modelMobileNet = tf.keras.models.load_model(('model/architecture_cnn_MobileNet_0.93.h5'),custom_objects={'KerasLayer':hub.KerasLayer})


print(modelMobileNet.summary())

# Path of selected image
#the_path = 'pictures/bell_tower/9660913378_67f17bb56e_m.jpg'


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
  print(f'Prediction: {labels[prediction_class].title()}')
  return labels[prediction_class].title()


#print(f"The prediction is: {the_prediction}")

@app.route('/')
def home():
	return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict():
  label_list = ['altar', 'apse', 'bell_tower', 'column', 'dome(inner)', 'dome(outer)', 'flying_buttress', 'gargoyle', 'stained_glass', 'vault']
  print(label_list)

  global COUNT
  img = request.files["picFile"]
  print(f"The selected picture is: {img}")
  
  img.save(f'uploads/{COUNT}.jpg')
  pic_path = f"uploads/{COUNT}.jpg"
  print(f"The picture path is: {pic_path}")
  # Predict with the MobileNet model
  the_prediction = classify_image(pic_path, modelMobileNet, label_list)
  COUNT += 1
  return render_template('predict.html', the_prediction = the_prediction)


@app.route('/load_img')
def load_img():

  global COUNT
  return send_from_directory('uploads', "{}.jpg".format(COUNT-1))


if __name__ == '__main__':
	app.run(debug=False)