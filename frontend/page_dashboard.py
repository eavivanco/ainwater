import streamlit as st
import toolkit_functions as tf

def app():
    st.header('Dashboard')
    
    # steps = st.slider("Cantidad de dias proyectados", 1, 30)
    # forecast = tf.get_forecast(steps)
        
    # tf.plot_timeserie(forecast)
    
    # st.subheader('Datos proyectados')
    # st.dataframe(forecast[forecast['is_pred'] ==1][['periodo', 'y']])
    
    # predictions = tf.predict(X)
    
    predictions = [1, 2, 3, 2]
        
    
    fi_class_predicted = predictions[0]
    fl1_class_predicted = predictions[1]
    fl2_class_predicted = predictions[2]
    fl3_class_predicted = predictions[3]
    
    st.subheader(f'The filamentous index of this image is: {fi_class_predicted}')
    st.subheader(f'The flóculo 1 of this image is: {fl1_class_predicted}')
    st.subheader(f'The flóculo 2 of this image is: {fl2_class_predicted}')
    st.subheader(f'The flóculo 3 of this image is: {fl3_class_predicted}')

    