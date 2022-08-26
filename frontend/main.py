import streamlit as st #framewotk for web app
import styles_config  #scrip for styles 
# import toolkit_functions as tf #library for toolkit functions
import page_inicio
import functions as fn
# import page_dashboard

styles_config.styles() #style for page

st.title("Cette merde fonctione")

with st.sidebar:
    st.image('images/logo.png', width=300)
    page = fn.menu_bar(orientation= 'vertical') #menu bar

if page == 'Inicio':
    page_inicio.app()

# if page == 'Dashboard':
#     page_dashboard.app()