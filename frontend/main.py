import streamlit as st #framewotk for web app
import streamlit_option_menu as sm #library for option menu
import styles_config  #scrip for styles 
import toolkit_functions as tf #library for toolkit functions
import page_inicio
import page_dashboard


styles_config.styles() #style for page

with st.sidebar:
    st.image('images/logo.png', width=200)
    page = tf.menu_bar(orientation= 'vertical') #menu bar

if page == 'Inicio':
    page_inicio.app()

if page == 'Dashboard':
    page_dashboard.app()
    