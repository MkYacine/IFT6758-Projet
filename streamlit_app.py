import streamlit as st
import pandas as pd
import numpy as np

"""
General template for your streamlit app. 
Feel free to experiment with layout and adding functionality!
Just make sure that the required functionality is included as well
"""

# Displaying the NHL Logo
col1, col2, col3 = st.columns(3)

with col1:
    st.write(' ')

with col2:
    st.image("Nhl_logo.png")

with col3:
    st.write(' ')

# Title of the App
st.markdown("<h1 style='text-align: center; color: black;'>Hockey Visualisation App</h1>", unsafe_allow_html=True)


with st.sidebar:
    # TODO: Add input for the sidebar
    st.sidebar.title('Model selection')

    workspace = st.text_input('Workspace', 'A11-Group')
    st.sidebar.selectbox('Model',('MLP','XGBoost', 'LogReg Distance', 'LogReg Distance & Angle'))
    version = st.text_input('Version', value = '1.0.0')

    if st.button('Get Model'): 
        pass



   

with st.container():
    # TODO: Add Game ID input
    user_input = st.text_area('GameID')

    st.write("Vous avez saisi :", user_input)

    st.button('Ping game')

with st.container():
    # TODO: Add Game info and predictions
    pass

with st.container():
    # TODO: Add data used for predictions
    pass

