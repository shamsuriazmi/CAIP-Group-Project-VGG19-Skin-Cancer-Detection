# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 03:17:37 2022

@author: Shamsuri Azmi
"""

import streamlit as st
import keras
from PIL import Image
from img_classification import teachable_machine_classification

st.title("Image Classification with Model Trained on Skin Cancer Dataset")
st.header("Skin Cancer Classification Example")
st.text("Upload a Skin Cancer Image for image classification as benign or malignant")

def load_model():
  model = keras.models.load_model('vgg_mb_final.h5')
  return model

model = load_model()

uploaded_file = st.file_uploader("Choose a Skin Image ...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Skin Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = teachable_machine_classification(image, model)
    # evaluation = scoring(image, 'vgg_mb_final.h5')
    if label == 0:
        st.write("The Skin Image is Benign")
    else:
        st.write("The Skin Image is Malignant")