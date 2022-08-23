import yaml
import pandas as pd
import numpy as np
import cv2 as cv
from joblib import load
import tensorflow as tf
from keras import models, layers
import json

### @app.put('/upload')
# def upload(periodo, total_boletas):
#     '''metodo api para la carga de datos en un archivo temporal de la aplicación'''
#     periodo, total_boletas = periodo.split(','), total_boletas.split(',')
#     tmp_tbl = {'periodo': periodo, 'total_boletas': [float(x) for x in total_boletas]}
#     f = open('tmp/tmp_tbl.yaml', 'w+')
#     yaml.dump(tmp_tbl, f, sort_keys=False)
#     f.close()
#     return 'OK'

def upload(image_data):
    '''metodo api para la carga de datos en un archivo temporal de la aplicación'''
    tmp_tbl = image_data
    f = open('tmp/tmp_tbl.yaml', 'w+')
    yaml.dump(tmp_tbl, f, sort_keys=False)
    f.close()
    return 'OK'

### @app.get('/datos_temporales') 
def datos_temporales():
    '''metodo para leer los datos temporales de la aplicación'''
    f = open('tmp/tmp_tbl.yaml', 'r')
    tmp_tbl = yaml.load(f, Loader=yaml.FullLoader)
    f.close()
    return tmp_tbl

### ETL para  @app.get('/forecast')
# def read_csv(path_csv = '', datetime = 'periodo'):
#     data = pd.read_csv(path_csv, parse_dates = [datetime])
#     if 'Unnamed: 0' in data.columns:
#         del data['Unnamed: 0']
#     return data  

# def time_variables(data, datetime = ''):
#     serie_dt = data[datetime]
#     holiday = read_csv('tmp/feriados_CL.csv', 'date')['date']
#     data['month'] = serie_dt.dt.month
#     data['dayofweek'] = serie_dt.dt.dayofweek
#     data['dayofmonth'] = serie_dt.dt.day
#     data['is_holiday'] = np.where(serie_dt.isin(holiday),1,0)
#     return data

# def data_asfreq(data, y= 'total_boletas', datetime = 'periodo', freq = 'd'):
#     data.set_index(datetime, inplace = True)
#     data = data.asfreq(freq)
#     data = data.sort_index()
#     if y != None:
#         data = data.rename(columns={y:'y'})
#     return data

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

# def df_tmp():
#     tmp_tbl = datos_temporales()
#     df_tmp = pd.DataFrame(tmp_tbl)
#     df_tmp['periodo'] = pd.to_datetime(tmp_tbl['periodo'])
#     df_tmp = df_tmp[['periodo', 'total_boletas']]
#     df_tmp = data_asfreq(df_tmp, y= 'total_boletas', datetime = 'periodo', freq = 'd')
#     return df_tmp

# def arr_tmp():
#     arr_tmp = np.ndarray(
#         shape=(1, 540, 720, 3),
#         dtype=np.uint8
#         )
#     tmp_tbl = datos_temporales()
#     arr_tmp = pd.DataFrame(tmp_tbl)
#     df_tmp['periodo'] = pd.to_datetime(tmp_tbl['periodo'])
#     df_tmp = df_tmp[['periodo', 'total_boletas']]
#     df_tmp = data_asfreq(df_tmp, y= 'total_boletas', datetime = 'periodo', freq = 'd')
#     return arr_tmp

    
# def etl_predict(last_window, steps):
#     last_window_y = last_window['y']
#     d_min = last_window.index.min()
#     cant_lags = len(last_window)
#     periods = cant_lags + steps
#     exog_tbl = pd.DataFrame()
#     exog_tbl['periodo'] = pd.date_range(start = d_min, periods = periods, freq = 'd')
#     exog_tbl = time_variables(exog_tbl, 'periodo')
#     exog_tbl = data_asfreq(exog_tbl, None, 'periodo', 'd')
#     return  last_window_y, exog_tbl

# def predict(steps):
#     model = load('model/forecaster.py')
#     last_window = df_tmp()
#     last_window_y, exog_tbl = etl_predict(last_window, steps)
#     pred = model.predict(steps=steps, 
#                          last_window= last_window_y,
#                          exog=exog_tbl)
#     return last_window, pred

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

### @app.get('/forecast')

# def forecast(steps):
#     name_dayofweek = {0: 'Lunes', 
#                       1: 'Martes',
#                       2: 'Miercoles', 
#                       3: 'Jueves', 
#                       4: 'Viernes', 
#                       5: 'Sabado', 
#                       6: 'Domingo'}
#     last_window, pred = predict(steps)
#     last_window['is_pred'] = 0
#     last_window.reset_index(inplace = True)

#     pred = pd.DataFrame(pred)
#     pred.rename(columns = {'pred': 'y'}, inplace = True)
#     pred.reset_index(inplace = True)
#     pred.rename(columns = {'index': 'periodo'}, inplace = True)
#     pred['is_pred'] = 1

#     df_timeserie = pd.concat([last_window, pred])
#     df_timeserie.sort_values('periodo', inplace = True)
#     df_timeserie.reset_index(inplace = True, drop = True)
#     df_timeserie = time_variables(df_timeserie, datetime = 'periodo')
#     df_timeserie['periodo'] = df_timeserie['periodo'].astype(str)
#     df_timeserie['day_name'] = df_timeserie['dayofweek'].map(lambda x: name_dayofweek[x])
#     return json.loads(df_timeserie.to_json())
