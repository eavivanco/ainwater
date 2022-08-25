from uuid import uuid4
import streamlit as st
import streamlit_option_menu as sm
import pandas as pd
import numpy as np
import cv2 as cv
import yaml
import requests
# import plotly
# import plotly.graph_objects as go
import tensorflow as tf
from keras import models, layers

### Cargar el esquema de la página
def read_schema_page():
    yaml_path= 'files/schema_page.yaml' 
    schema_page=yaml.load(open(yaml_path),Loader=yaml.FullLoader)
    return schema_page

### Menu de la pagina principal
def menu_bar(select_color="#0d63ba",orientation="horizontal"):
    schema = read_schema_page()
    menu_list = [i for i in schema.keys()]
    menu_icon = [schema[i] for i in menu_list] #https://icons.getbootstrap.com
    page = sm.option_menu(None, menu_list, 
        icons=menu_icon, 
        menu_icon="cast", default_index=0, orientation=orientation,
        styles={"nav-link-selected": {"background-color": select_color}})
    return page

def load_url():
    url_path= 'files/url.yaml' 
    url=yaml.load(open(url_path),Loader=yaml.FullLoader)
    return url['url']

### ---- ETL ---- ###

def get_Xtf(img_file):
    '''
    Transform images with true fast contours to an array from image information
    '''
    i = 0
    arr_images = np.ndarray(
        shape=(1, 540, 720, 3),
        dtype=np.uint8
        )
    pre_img = cv.imread(img_file)
    if pre_img.shape[0] > 99:
        img = add_fast_true(pre_img) 
        arr_images[i] = img / 255
        i += 1
    return arr_images

def add_fast_true(pre_img):
  fast = cv.FastFeatureDetector_create()
  kp = fast.detect(pre_img, None)
  img = cv.drawKeypoints(pre_img, kp, None, color=(255,0,0))
  return img

def load_image(image_file):
	img = cv.imread(image_file)
	return img

### ---- OLD ETL ---- ###

# def read_csv(path_csv = '', datetime = ''):
#     data = pd.read_csv(path_csv, parse_dates = [datetime])
#     if 'Unnamed: 0' in data.columns:
#         del data['Unnamed: 0']
#     return data

### ---- API ---- ###
# def send_data(data):
#     '''guardar los datos cargados en backend'''
#     periodo = data['periodo'].astype(str)
#     periodo = ','.join(periodo)
#     total_boletas = data['total_boletas'].astype(str)
#     total_boletas = ','.join(total_boletas)
    
#     url =  load_url()
#     method = 'upload'
#     put_api = f'{url}/{method}?periodo={periodo}&total_boletas={total_boletas}'
#     response = requests.put(put_api)
#     return response.status_code

# def send_data():
#     '''guardar los datos cargados en backend'''
#     image_id = uuid4()
    
#     url =  load_url()
#     method = 'upload'
#     put_api = f'{url}/{method}?id={image_id}'
#     response = requests.put(put_api)
#     return response.status_code

# def get_data():
#     '''obtener los datos cargados en backend'''
#     url =  load_url()
#     method = 'datos_temporales'
#     get_api = f'{url}/{method}'
#     response = requests.get(get_api)
#     return response.json()

# def get_forecast(steps):
#     '''obtener los datos cargados en backend'''
#     url =  load_url()
#     method = 'forecast'
#     get_api = f'{url}/{method}?steps={steps}'
#     response = requests.get(get_api)
#     forecast = response.json()
#     forecast = pd.DataFrame(forecast)
#     forecast['periodo'] = forecast['periodo'].astype('datetime64[ns]')
#     forecast['y'] = forecast['y'].astype(float)
#     return forecast

# def get_prediction(image_name):
#     '''obtener los datos cargados en backend'''
#     url =  load_url()
#     method = 'prediction'
#     get_api = f'{url}/{method}?image={image_name}'
#     response = requests.get(get_api)
#     prediction = response.json()
#     prediction = pd.DataFrame(prediction)

#     return prediction

### ---- BACK ---- ###

def create_sequential(IMG_SIZE, NB_OUTPUTS, LR, NAME):
    h, w = IMG_SIZE[0], IMG_SIZE[1]
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(h, w, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, name=f'dense1_{NAME}'))
    model.add(layers.Dense(2 * NB_OUTPUTS, name=f'dense2_{NAME}'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    return model

def predict(X):
    predictions = {
        'fi': 0,
        'fl1': 0,
        'fl2': 0,
        'fl3': 0
    }
    IMG_SIZE = [540, 720]
    LR = 0.001
    fi_model = create_sequential(IMG_SIZE, 6, LR, NAME='FI')
    fi_model.load_weights('./weights/FilamentousIndex_ghmodelweights.h5')
    fl1_model = create_sequential(IMG_SIZE, 2, LR, NAME='FL1')
    fl1_model.load_weights('./weights/Floculo1_ghmodelweights.h5')
    fl2_model = create_sequential(IMG_SIZE, 2, LR, NAME='FL2')
    fl2_model.load_weights('./weights/Floculo2_ghmodelweights.h5')
    fl3_model = create_sequential(IMG_SIZE, 2, LR, NAME='FL3')
    fl3_model.load_weights('./weights/Floculo3_ghmodelweights.h5')
    
    fi_pre_prediction = fi_model.predict(X)
    predictions['fi'] = np.argmax(fi_pre_prediction,axis=1)[0]
    fl1_pre_prediction = fl1_model.predict(X)
    predictions['fl1'] = np.argmax(fl1_pre_prediction,axis=1)[0]
    fl2_pre_prediction = fl2_model.predict(X)
    predictions['fl2'] = np.argmax(fl2_pre_prediction,axis=1)[0]
    fl3_pre_prediction = fl3_model.predict(X)
    predictions['fl3'] = np.argmax(fl3_pre_prediction,axis=1)[0]

    return predictions

### ---- FIGURES ---- ###

# def plot_timeserie(forecast):
#     fill_real = forecast[forecast['is_pred'] ==0]
#     fill_pred = forecast[len(fill_real)-1:]
#     fill_holiday = forecast[forecast['is_holiday'] ==1]
    
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=fill_real['periodo'], 
#                              y=fill_real['y'], 
#                              fill = 'tozeroy',
#                              mode='lines', 
#                              line = {'dash': 'solid'},
#                              name='Real'))
    
#     fig.add_trace(go.Scatter(x=fill_pred['periodo'], 
#                              y=fill_pred['y'], 
#                              fill = 'tozeroy',
#                              mode='lines', 
#                              line = {'dash': 'dash'},
#                              name='Forecast'))
    
#     fig.add_trace(go.Scatter(x=fill_holiday['periodo'],
#                              y=fill_holiday['y'],
#                              mode ='markers',
#                              name='feriados'))
                                
    
#     st.plotly_chart(fig)
