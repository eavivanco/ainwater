import streamlit as st
import streamlit_option_menu as sm
import pandas as pd
import yaml
import requests
import plotly
import plotly.graph_objects as go

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


def read_csv(path_csv = '', datetime = ''):
    data = pd.read_csv(path_csv, parse_dates = [datetime])
    if 'Unnamed: 0' in data.columns:
        del data['Unnamed: 0']
    return data

def load_url():
    url_path= 'files/url.yaml' 
    url=yaml.load(open(url_path),Loader=yaml.FullLoader)
    return url['url']


### ---- API ---- ###
def send_data(data):
    '''guardar los datos cargados en backend'''
    periodo = data['periodo'].astype(str)
    periodo = ','.join(periodo)
    total_boletas = data['total_boletas'].astype(str)
    total_boletas = ','.join(total_boletas)
    
    url =  load_url()
    method = 'upload'
    put_api = f'{url}/{method}?periodo={periodo}&total_boletas={total_boletas}'
    response = requests.put(put_api)
    return response.status_code

def get_data():
    '''obtener los datos cargados en backend'''
    url =  load_url()
    method = 'datos_temporales'
    get_api = f'{url}/{method}'
    response = requests.get(get_api)
    return response.json()

def get_forecast(steps):
    '''obtener los datos cargados en backend'''
    url =  load_url()
    method = 'forecast'
    get_api = f'{url}/{method}?steps={steps}'
    response = requests.get(get_api)
    forecast = response.json()
    forecast = pd.DataFrame(forecast)
    forecast['periodo'] = forecast['periodo'].astype('datetime64[ns]')
    forecast['y'] = forecast['y'].astype(float)
    return forecast

### ---- FIGURES ---- ###

def plot_timeserie(forecast):
    fill_real = forecast[forecast['is_pred'] ==0]
    fill_pred = forecast[len(fill_real)-1:]
    fill_holiday = forecast[forecast['is_holiday'] ==1]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fill_real['periodo'], 
                             y=fill_real['y'], 
                             fill = 'tozeroy',
                             mode='lines', 
                             line = {'dash': 'solid'},
                             name='Real'))
    
    fig.add_trace(go.Scatter(x=fill_pred['periodo'], 
                             y=fill_pred['y'], 
                             fill = 'tozeroy',
                             mode='lines', 
                             line = {'dash': 'dash'},
                             name='Forecast'))
    
    fig.add_trace(go.Scatter(x=fill_holiday['periodo'],
                             y=fill_holiday['y'],
                             mode ='markers',
                             name='feriados'))
                                
    
    st.plotly_chart(fig)
