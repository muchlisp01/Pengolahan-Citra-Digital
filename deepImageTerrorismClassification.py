import tensorflow as tf
import numpy as np
from tensorflow import keras
import base64
from PIL import Image
import io

def loadModel():
     model =  keras.models.load_model('./models/terrorism-AllLayer-weights.11-0.04-0.99-0.98.h5')
     return model

model = loadModel()

def predictImageTerrorism(base64_string = str(), model = model):
    images = base64.b64decode(base64_string)
    images= Image.open(io.BytesIO(images))
    images = images.resize((224,224))
    images = tf.keras.preprocessing.image.img_to_array(images)
    images = np.expand_dims(images, axis=0)
    prediction = model.predict(images).flatten()
    predictions = tf.nn.sigmoid(prediction)
    result = tf.where(predictions < 0.5, 0, 1).numpy()[0]
    if result == 0:
        finalResult = 'nonterrorism'
        confidence = round(100 - (100*(predictions.numpy()[0])), 2)
    else:
        finalResult = 'terrorism'
        confidence = round(100*(predictions.numpy()[0]), 2)
    
    return {'confidenceTerrorism': confidence, 'resultTerrorism': finalResult}
    