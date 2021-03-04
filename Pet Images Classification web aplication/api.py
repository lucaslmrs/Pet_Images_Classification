from ast import Str
import os
from flask import Flask
from flask import request
from flask import render_template
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

app = Flask(__name__, template_folder="templates")
UPLOAD_FOLDER = "/Users/lucas vinicios/Desktop/Pet Images Classification web aplication"


def predict(PATH_IMAGE):
    images = list()
    img = Image.open(PATH_IMAGE)
    img = np.asarray(img.resize((128,128)))
    rgb_weights = [0.2989, 0.5870, 0.1140]
    img = np.dot(img[...,:3], rgb_weights)
    images.append(img)

    model = tf.keras.models.load_model(UPLOAD_FOLDER + "/models")
    resultado = model.predict(np.array(images).reshape(1,128,128,1))[0]
    
    if resultado[0]> 0.5: return f"Tem {np.round(resultado[0]*100, 2)}% de chance de ser um gato, predict: {resultado}"
    else: return f"Tem {np.round(resultado[1]*100 , 2)}% de chance de ser um cachorro predict: {resultado}"

@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            image_location = os.path.join(
                UPLOAD_FOLDER + "/images",
                image_file.filename
            )
            image_file.save(image_location)

            return render_template("index.html", prediction=predict(image_location))
    return render_template("index.html", prediction=0)

if __name__ == "__main__":
    app.run(port=8888, debug=True)

