import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
# Flask utils
from flask import Flask, redirect, url_for, request, render_template

# Define a flask app
app = Flask(__name__)


def fruit_predict(file_path):
    model = tf.keras.models.load_model(r'/content/drive/MyDrive/Dataset Plant Disease/fruit.h5')
    test_datagen_1=ImageDataGenerator(rescale=1)
    test_generator_1=test_datagen_1.flow_from_directory(
    test_dir,
    target_size=(128,128),
    batch_size=20,
    class_mode='categorical')
    img=image.load_img(file_path,target_size=(128,128))
    x=image.img_to_array(img)
    x=np.expand_dims(x,axis=0)
    y=np.argmax(model.predict(x),axis=1)
    index=['Apple___Black_rot', 'Apple___healthy', 'Corn_(maize)___healthy', 'Corn_(maize)___Northern_Leaf_Blight', 'Peach___Bacterial_spot', 'Peach___healthy']
    return index[y[0]]


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        # Make prediction
        preds = fruit_predict(file_path)
        #result=preds
        #return result
        return render_template('index.html',result=preds)
    

if __name__ == '__main__':
    app.run(port=5001,debug=True)
