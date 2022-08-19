import streamlit as st
from PIL import Image


def styles():
    page_config()
    footer()
    pag_style()


def page_config():
    favicon = Image.open('images/favicon.ico')
    st.set_page_config(page_title='demo deploy', 
                        page_icon=favicon)

def footer():
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            footer:after {
                content:'powered by cc'; 
	            visibility: visible;
	            display: block;
	            position: relative;
	            #background-color: red;
	            padding: 5px;
	            top: 2px;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 


def pag_style():
    padding_top=1
    st.markdown(
        f"""
<style>
    .appview-container .main .block-container{{
        padding-top: {padding_top}rem;    
        }}
</style>
""",
        unsafe_allow_html=True,
    )




