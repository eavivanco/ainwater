import streamlit as st
import toolkit_functions as tf
import pandas as pd
import cv2 as cv

def app():
    st.header('Machine Learning model for image recognition')
    st.write('This app was made to ')
    st.write('Para empezar, cargue un archivo csv con los datos de las boletas del periodo que desea analizar')
    
    #formulario para cargar una imagen
    with st.form(key = 'upload_data',
                 clear_on_submit=False):
        
        file = st.file_uploader(label='image_uploader',
                                type=['png', 'jpg'],
                                accept_multiple_files=False) #cargar archivo
        save = st.form_submit_button('Cargar') #enviar datos a api backend
        
    if save:
        try:
            with st.spinner('cargando datos'):
                data = tf.read_csv(file, 'periodo')
                status_code = tf.send_data(data)
                if status_code == 200:
                    st.success('Datos cargados correctamente')
                    #st.balloons()
        except ValueError:
            st.error('No se ha cargado ningun archivo')
            
            
    with st.expander('ver ultimos datos cargados'):  
        data_tmp = tf.get_data()
        st.image(data_tmp)
        # data_tmp = pd.DataFrame(data_tmp)
        # st.dataframe(data_tmp)
        # st.info('Para ver el dashboard, seleccione la opción "Dashboard"')
        