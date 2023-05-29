import streamlit as st
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os 
import ydata_profiling
from ydata_profiling import ProfileReport
from operator import index

import plotly.express as px
from pycaret.regression import setup, compare_models, pull, save_model, load_model



if os.path.exists('./insurance_dataset.csv'): 
    df = pd.read_csv('insurance_dataset.csv', index_col=None)



with st.sidebar: 
    st.title("Auto_ML_webapp")
    st.image("https://editor.analyticsvidhya.com/uploads/77447AutoML.png")
    st.info("This application automates the task of data exploration and build the model for the dataset.")
    choice = st.radio("Navigation", ["Upload","Profiling","Modelling", "Download"])
    


if choice == "Upload":
    st.title("Upload the Dataset")
    file = st.file_uploader("Upload the Dataset")
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('insurance_dataset.csv', index=None)
        st.dataframe(df)



if choice == "Profiling": 
    st.title("Exploratory Data Analysis")
    profile_df = df.profile_report()
    st_profile_report(profile_df)

if choice == "Modelling": 
    chosen_target = st.selectbox('Choose the target Column', df.columns)
    if st.button('Run Modelling'): 
        setup(df, target=chosen_target, silent=True)
        setup_df = pull()
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.dataframe(compare_df)
        save_model(best_model, 'best_model')


if choice == "Download": 
    with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="best_model.pkl")