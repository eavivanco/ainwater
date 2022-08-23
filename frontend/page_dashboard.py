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
    predictions = {
        'fi': 1,
        'fl1': 1,
        'fl2': 2,
        'fl3': 1
    }
    fi_class_predicted = predictions['fi']
    fl1_class_predicted = predictions['fl1']
    fl2_class_predicted = predictions['fl2']
    fl3_class_predicted = predictions['fl3']
    
    st.subheader(f'The filamentous index of this image is: {fi_class_predicted}')
    st.subheader(f'The flóculo 1 of this image is: {fl1_class_predicted}')
    st.subheader(f'The flóculo 2 of this image is: {fl2_class_predicted}')
    st.subheader(f'The flóculo 3 of this image is: {fl3_class_predicted}')

    