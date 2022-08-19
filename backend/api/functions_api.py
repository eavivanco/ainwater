import yaml
import pandas as pd
import numpy as np
from joblib import load
import json

### @app.put('/upload')
def upload(periodo, total_boletas):
    '''metodo api para la carga de datos en un archivo temporal de la aplicación'''
    periodo, total_boletas = periodo.split(','), total_boletas.split(',')
    tmp_tbl = {'periodo': periodo, 'total_boletas': [float(x) for x in total_boletas]}
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
def read_csv(path_csv = '', datetime = 'periodo'):
    data = pd.read_csv(path_csv, parse_dates = [datetime])
    if 'Unnamed: 0' in data.columns:
        del data['Unnamed: 0']
    return data  

def time_variables(data, datetime = ''):
    serie_dt = data[datetime]
    holiday = read_csv('tmp/feriados_CL.csv', 'date')['date']
    data['month'] = serie_dt.dt.month
    data['dayofweek'] = serie_dt.dt.dayofweek
    data['dayofmonth'] = serie_dt.dt.day
    data['is_holiday'] = np.where(serie_dt.isin(holiday),1,0)
    return data

def data_asfreq(data, y= 'total_boletas', datetime = 'periodo', freq = 'd'):
    data.set_index(datetime, inplace = True)
    data = data.asfreq(freq)
    data = data.sort_index()
    if y != None:
        data = data.rename(columns={y:'y'})
    return data

def df_tmp():
    tmp_tbl = datos_temporales()
    df_tmp = pd.DataFrame(tmp_tbl)
    df_tmp['periodo'] = pd.to_datetime(tmp_tbl['periodo'])
    df_tmp = df_tmp[['periodo', 'total_boletas']]
    df_tmp = data_asfreq(df_tmp, y= 'total_boletas', datetime = 'periodo', freq = 'd')
    return df_tmp

    
def etl_predict(last_window, steps):
    last_window_y = last_window['y']
    d_min = last_window.index.min()
    cant_lags = len(last_window)
    periods = cant_lags + steps
    exog_tbl = pd.DataFrame()
    exog_tbl['periodo'] = pd.date_range(start = d_min, periods = periods, freq = 'd')
    exog_tbl = time_variables(exog_tbl, 'periodo')
    exog_tbl = data_asfreq(exog_tbl, None, 'periodo', 'd')
    return  last_window_y, exog_tbl

def predict(steps):
    model = load('model/forecaster.py')
    last_window = df_tmp()
    last_window_y, exog_tbl = etl_predict(last_window, steps)
    pred = model.predict(steps=steps, 
                         last_window= last_window_y,
                         exog=exog_tbl)
    return last_window, pred

### @app.get('/forecast')
def forecast(steps):
    name_dayofweek = {0: 'Lunes', 
                      1: 'Martes',
                      2: 'Miercoles', 
                      3: 'Jueves', 
                      4: 'Viernes', 
                      5: 'Sabado', 
                      6: 'Domingo'}
    last_window, pred = predict(steps)
    last_window['is_pred'] = 0
    last_window.reset_index(inplace = True)

    pred = pd.DataFrame(pred)
    pred.rename(columns = {'pred': 'y'}, inplace = True)
    pred.reset_index(inplace = True)
    pred.rename(columns = {'index': 'periodo'}, inplace = True)
    pred['is_pred'] = 1

    df_timeserie = pd.concat([last_window, pred])
    df_timeserie.sort_values('periodo', inplace = True)
    df_timeserie.reset_index(inplace = True, drop = True)
    df_timeserie = time_variables(df_timeserie, datetime = 'periodo')
    df_timeserie['periodo'] = df_timeserie['periodo'].astype(str)
    df_timeserie['day_name'] = df_timeserie['dayofweek'].map(lambda x: name_dayofweek[x])
    return json.loads(df_timeserie.to_json())
