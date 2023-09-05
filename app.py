from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adamax
from PIL import Image

app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello_world():
    return render_template("index.html")

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "static/" + imagefile.filename
    imagefile.save(image_path)

    image = Image.open(image_path)
    loaded_model = tf.keras.models.load_model('./bestmodel.h5', compile=False)
    loaded_model.compile(Adamax(learning_rate= 0.001), loss= 'categorical_crossentropy', metrics= ['accuracy'])

    # Preprocess the image
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    # Make predictions
    classes = ['Covid', 'Normal', 'Viral Pneumonia']
    predictions = loaded_model.predict(img_array)
    class_labels = classes
    score = tf.nn.softmax(predictions[0])

    result = f"Prediction: {class_labels[tf.argmax(score)]}, confidence: {np.max(score.numpy()):.2f}"


    return render_template("index.html", prediction=result, img_path=image_path)


if __name__ == "__main__":
    app.run(port=3000, debug=True)
