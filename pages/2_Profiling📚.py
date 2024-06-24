import streamlit as st
import os
import pandas as pd
from st_pages import Page, show_pages, add_page_title

st.set_page_config(
    page_title="Auto ML and DS",
    page_icon="https://cdn-icons-png.flaticon.com/128/4616/4616734.png",
    layout="wide",
    initial_sidebar_state="auto",
)

with st.sidebar:
    # Place the show_pages function here
 

    st.sidebar.write("Developed by Yazeed")

if os.path.exists("./dataset.csv"):
    df = pd.read_csv("dataset.csv", index_col=None)

st.title("Exploratory Data Analysis")
st.write("Is there null values?")
st.write(df.isna().sum().any())
if df.isna().sum().any():
        nullValues = st.radio("Drop them or fill with mean", ["", "drop", "mean"])
        if nullValues == "":
            st.write("Please select an option.")
        if nullValues == "drop":
            df = df.dropna()
            st.success("null values were dropped!")
        if nullValues == "mean":
            df.fillna(df.mean(numeric_only=True), inplace=True)
            st.success("null values were filled with mean!")
st.write("Is there duplicated?")
st.write(df.duplicated().sum().any())
if df.duplicated().sum().any():
        st.write("It will be dropped!")
        df = df.drop_duplicates()
        st.success("duplicates were dropped!")

st.subheader("Now let's showcase some EDA")
st.write("Showcasing the first 5 rows")
st.dataframe(df.head())
st.write("Summary of the dataset")
st.write(df.describe())
    

df.to_csv("dataset.csv", index=None)