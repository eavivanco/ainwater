import streamlit as st
import toolkit_functions as tf
import pandas as pd
import cv2 as cv

def app():
    st.header('Machine Learning model for image recognition')
    st.write('This app was made to recognize different components inside a microscopic image')
    st.write('To start, please upload a photo')
    
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
                # st.write(f'This is the file received : {file.name}')
                # posted_image = cv.imread(file)
                # st.write(f'This is the image procesed received : {posted_image}')
                # st.image(posted_image)
                # image_data = tf.get_Xtf(file)
                status_code = tf.send_data()
                if status_code == 200:
                    st.success('Datos cargados correctamente')
                    #st.balloons()
        except ValueError:
            st.error('No se ha cargado ningun archivo')
            
            
    with st.expander('ver ultimos datos cargados'):  
        data_tmp = tf.get_data()
        # data_tmp = pd.DataFrame(data_tmp)
        # st.dataframe(data_tmp)
        # st.info('Para ver el dashboard, seleccione la opción "Dashboard"')
        