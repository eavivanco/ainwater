from functions import get_Xtf, create_sequential

from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
import tensorflow as tf
import keras
import numpy as np
import time

IMG_PATH = './images'
IMG_SIZE = [540, 720]
LR = 0.001

st.title('Microscopic images recognition by Ainwater')
image_number = st.slider("Choose the image you want to see", 1, 5, 3)
st.image(f'./images/{image_number}.png')

with st.spinner(text='In progress'):
    time.sleep(1)
    # st.success('Done')
    
X = get_Xtf(IMG_PATH, image_number)

# fi_model = tf.keras.models.load_model('/Users/estebanvivanco/Desktop/models/FilamentousIndex_ghmodel.h5')
fi_model = create_sequential(IMG_SIZE, 6, LR, NAME='FI')
fi_model.load_weights('/Users/estebanvivanco/Desktop/weights/FilamentousIndex_ghmodelweights.h5')
fi_pre_prediction = fi_model.predict(X)
fi_class_predicted = np.argmax(fi_pre_prediction,axis=1)[0]

# fl1_model = tf.keras.models.load_model('/Users/estebanvivanco/Desktop/models/Floculo1_ghmodel.h5')
fl1_model = create_sequential(IMG_SIZE, 2, LR, NAME='FL1')
fl1_model.load_weights('/Users/estebanvivanco/Desktop/weights/Floculo1_ghmodelweights.h5')
fl1_pre_prediction = fl1_model.predict(X)
fl1_class_predicted = np.argmax(fl1_pre_prediction,axis=1)[0]

# fl2_model = tf.keras.models.load_model('/Users/estebanvivanco/Desktop/models/Floculo2_ghmodel.h5')
fl2_model = create_sequential(IMG_SIZE, 2, LR, NAME='FL2')
fl2_model.load_weights('/Users/estebanvivanco/Desktop/weights/Floculo2_ghmodelweights.h5')
fl2_pre_prediction = fl2_model.predict(X)
fl2_class_predicted = np.argmax(fl2_pre_prediction,axis=1)[0]

# fl3_model = tf.keras.models.load_model('/Users/estebanvivanco/Desktop/models/Floculo3_ghmodel.h5')
fl3_model = create_sequential(IMG_SIZE, 2, LR, NAME='FL3')
fl3_model.load_weights('/Users/estebanvivanco/Desktop/weights/Floculo3_ghmodelweights.h5')
fl3_pre_prediction = fl3_model.predict(X)
fl3_class_predicted = np.argmax(fl3_pre_prediction,axis=1)[0]

st.subheader(f'The filamentous index of this image is: {fi_class_predicted}')
st.subheader(f'The flóculo 1 of this image is: {fl1_class_predicted}')
st.subheader(f'The flóculo 2 of this image is: {fl2_class_predicted}')
st.subheader(f'The flóculo 3 of this image is: {fl3_class_predicted}')


