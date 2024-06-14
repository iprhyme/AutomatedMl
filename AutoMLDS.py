import streamlit as st
from sklearn.preprocessing import LabelEncoder
import pandas as pd


import os
from st_pages import Page, show_pages

st.set_option("deprecation.showPyplotGlobalUse", False)
st.set_page_config(
    page_title="Auto ML and DS",
    page_icon="https://cdn-icons-png.flaticon.com/128/4616/4616734.png",
    layout="wide",
    initial_sidebar_state="auto",
)
# Check if the dataset file exists
if os.path.exists("./dataset.csv"):
    df = pd.read_csv("dataset.csv", index_col=None)

# Initialize session state for chosen_target if not already set
if "chosen_target" not in st.session_state:
    st.session_state.chosen_target = ""
if "classification" not in st.session_state:
    st.session_state.classification = True
if "chosen_model" not in st.session_state:
    st.session_state.chosen_model = ""

# Sidebar for navigation
with st.sidebar:
    # Place the show_pages function here
    show_pages(
        [
            Page("Home.py", "Home", "🏠",in_section=False),
            Page("AutoMLDS.py", "Upload", "📲",in_section=False),
            Page("Profiling.py", "Profiling", ":📚:",in_section=False),
            Page("Visual.py", "Visualization", "📊",in_section=False),
            Page("Modeling.py", "Modeling", "⚙️",in_section=False),
        ]
    )

    st.sidebar.write("Developed by Yazeed")
selected_columns = ""

# Upload section

st.title("Upload Your Dataset")
file = st.file_uploader("Upload Your Dataset", type="csv")
data_choice = st.checkbox("Use an example dataset?")
if (file and not data_choice) or (not file and data_choice):
    if file:
        df = pd.read_csv(file, index_col=None)
        st.dataframe(df)
    elif data_choice:

        df = pd.read_csv(
            "insurance.csv",
            index_col=None,
        )

        st.dataframe(df)
    with st.expander("🧑‍🔬 Important settings!"):

        model_type = st.radio(
            "",
            ["Classification", "Regression"],
            captions=[
                "for predicting categorical values (ex. yes,no)",
                "for predicting continuous values (ex. price)",
            ],
        )
        if model_type == "Regression":
            st.session_state.classification = False
        if model_type == "Classification":
            st.session_state.classification = True

        # dropping columns option
        selectedNA_columns = st.multiselect(
            "Select the Columns you want to drop", df.columns.tolist()
        )
        if selectedNA_columns:
            df = df.drop(selectedNA_columns, axis=1)

        # Target column

        st.info(
            "it's important to choose the feature that you want to predict correctly."
        )
        st.session_state.chosen_target = st.selectbox(
            "Choose the Target Column",
            df.columns,
            index=(
                df.columns.get_loc(st.session_state.chosen_target)
                if st.session_state.chosen_target in df.columns
                else 0
            ),
        )
        st.write("after converting the non-numerical to numerical")
        st.write("let's show the first 5 rows of the updated dataset")
        objects_left=df.select_dtypes(include=['object']).columns.tolist()
        le = LabelEncoder()
        for col in objects_left:
            
            df[col] = le.fit_transform(df[col])
         
        st.dataframe(df.head())
            
                
        
        df.to_csv("dataset.csv", index=None)
else:
    st.warning("Please choose only one")
