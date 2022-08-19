import streamlit as st
import toolkit_functions as tf

def app():
    st.header('Dashboard')
    
    steps = st.slider("Cantidad de dias proyectados", 1, 30)
    forecast = tf.get_forecast(steps)
        
    tf.plot_timeserie(forecast)
    
    st.subheader('Datos proyectados')
    st.dataframe(forecast[forecast['is_pred'] ==1][['periodo', 'y']])

    