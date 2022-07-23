import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from dash import html
from dash import dcc


global df

st.set_page_config(layout = "wide")
def show_visualisation():
    components.iframe("https://app.powerbi.com/reportEmbed?reportId=afadf8e8-31cc-4aef-8690-cd7c26edfdad&autoAuth=true&ctid=bceba7fc-937d-449b-bec8-d1cf87b0c6c7&config=eyJjbHVzdGVyVXJsIjoiaHR0cHM6Ly93YWJpLW5vcnRoLWV1cm9wZS1yZWRpcmVjdC5hbmFseXNpcy53aW5kb3dzLm5ldC8ifQ%3D%3D", height=500)
