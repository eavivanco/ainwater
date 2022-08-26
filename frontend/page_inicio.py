import streamlit as st
# import toolkit_functions as tf
# import pandas as pd
import functions as fn
import cv2 as cv
from PIL import Image

def app():
    """
    `app()` is the main function of the app. It contains the main logic of the app
    """
    st.header('Machine Learning model for image recognition')
    st.write('This app was made to recognize different components inside a microscopic image')
    st.write('To start, please upload a photo')
    
    #formulario para cargar una imagen
    with st.form(key = 'upload_data',
                 clear_on_submit=False):
        
        image_file = st.file_uploader(
            label='image_uploader',      
            type=['png', 'jpg'],
            accept_multiple_files=False) #cargar archivo
        save = st.form_submit_button('Cargar') #enviar datos a api backend
    
    if save:
        try:
            with st.spinner('cargando datos'):
                st.subheader('Original photo')
                st.image(image_file)
                image = fn.load_image(image_file)
                st.write(f'Original size: {image.size}')
                
                
                st.subheader('Resized photo')
                resized_image = fn.resize_image(image)
                st.image(resized_image)
                st.write(f'New size: {resized_image.size}')
                
                st.subheader('True fast photo')
                X = fn.get_Xtf(resized_image)
                processed_image = Image.fromarray(X[0])
                st.image(processed_image)
                
                predictions = [1,2,3,4]
                fi_cp, fl1_cp, fl2_cp, fl3_cp = predictions
                st.write(f'The filamentous index of this image is: {fi_cp}')
                st.write(f'The flóculo 1 of this image is: {fl1_cp}')
                st.write(f'The flóculo 2 of this image is: {fl2_cp}')
                st.write(f'The flóculo 3 of this image is: {fl3_cp}')
        except ValueError:
            st.error('No se ha cargado ningun archivo')
            
    # if save:
    #     try:
    #         with st.spinner('cargando datos'):
    #             st.write(f'This is the file received : {file.name}')
    #             posted_image = cv.imread(file)
    #             st.write(f'This is the image procesed received : {posted_image}')
    #             st.image(posted_image)
    #             image_data = tf.get_Xtf(file)
    #             status_code = tf.send_data()
    #             # ---- WITH API ----
    #             if status_code == 200:
    #                 st.success('Datos cargados correctamente')
    #                 #st.balloons()
    #     except ValueError:
    #         st.error('No se ha cargado ningun archivo')
            
            
    # with st.expander('ver ultimos datos cargados'):  
    #     data_tmp = tf.get_data()
    #     data_tmp = pd.DataFrame(data_tmp)
    #     st.dataframe(data_tmp)
    #     st.info('Para ver el dashboard, seleccione la opción "Dashboard"')
        