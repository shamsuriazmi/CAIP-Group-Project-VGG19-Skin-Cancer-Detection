# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 03:28:53 2022

@author: Fatihah
"""

import keras
import streamlit as st
from PIL import Image, ImageOps
import numpy as np

@st.cache(allow_output_mutation=True)
def teachable_machine_classification(img, vgg_mb_final):
    # Load the model
    model = keras.models.load_model('vgg_mb_final.h5')

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = img
    #image sizing
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    
    return np.argmax(prediction)  # return position of the highest probability

# def scoring(img, mod):
#     # Load the model
#     model = keras.models.load_model('vgg_mb_final.h5')

#     # Create the array of the right shape to feed into the keras model
#     data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
#     image = img
#     #image sizing
#     size = (224, 224)
#     image = ImageOps.fit(image, size, Image.ANTIALIAS)

#     #turn the image into a numpy array
#     image_array = np.asarray(image)
#     # Normalize the image
#     normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

#     # Load the image into the array
#     data[0] = normalized_image_array

#     # run the inference
#     score = model.evaluate(data)
    
#     return score  

