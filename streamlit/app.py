import streamlit as st
from Dashboard_page import show_visualisation
from model_page import show_model
from PIL import Image


menu = ["Home", "Dashboard", "Modelling"]
st.sidebar.header('Menu')
page = st.sidebar.selectbox('Select a Menu',menu)

if page =='Home':
    image = Image.open('bg.jpg')
    #st.image(image, width= 1098)
    col1, col2, col3 = st.columns([2,0.5,3])
    with col1:
        st.header("CUSTOMER CHURN PREDICTION APLLICATION")
        with st.expander("Overview"):
          st.write(""" This is a macchine learning application is built to predict
        Customer Churn in Telecommunication Industry.
        Using this application will reduce Churn rate and enable the comapany
        to discover and adress custormer who is about to churn.
        **Sample of data to be upload in this application is made available in the test folder.**


        """)
    with col3:
        video_file = open('preview.WEBM', 'rb')
        video_preview = video_file.read()

        st.video(video_preview)






elif page=='Dashboard':
    show_visualisation()

else:
    show_model()
