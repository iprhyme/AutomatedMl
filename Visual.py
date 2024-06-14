
import streamlit as st  
import plotly.express as px 
import pandas as pd
import matplotlib as plt
from st_pages import Page, show_pages, add_page_title
import os

st.set_page_config(
    page_title="Auto ML and DS",
    page_icon="https://cdn-icons-png.flaticon.com/128/4616/4616734.png",
    layout="wide",
    initial_sidebar_state="auto",
)

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

if os.path.exists("./dataset.csv"):
    df = pd.read_csv("dataset.csv", index_col=None)

st.title("Visualization!!")

# corr
try:
    corr = df.corr()
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu",
        title="Correlation Heatmap",
    )
    st.plotly_chart(fig)
except Exception as e:
    st.write("error during the correlation matrix, make sure you choose everything correct")


# pie chart
if st.session_state.classification:
    st.subheader("Distribution of " + st.session_state.chosen_target + " Variable")
    target_counts = df[st.session_state.chosen_target].value_counts()
    labels = df[st.session_state.chosen_target].unique()
    figure = px.pie(
        names=labels,
        values=target_counts,
        color_discrete_sequence=["#A6B1E1", "#424874", "#27374D", "#9DB2BF"],
    )
    st.plotly_chart(figure)

st.write("#### Customizable Scatter Plot")
st.markdown("Select columns to plot:")
x_col = st.selectbox("X-axis", df.columns)
y_col = st.selectbox("Y-axis", df.columns)
fig = px.scatter(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
st.plotly_chart(fig)

st.write("#### Customizable Bar Plot")
st.markdown("Select columns to plot:")
x_col = st.selectbox("Select Feature", df.columns, key="x_axis")
y_col = df[x_col].value_counts().index
y_values = df[x_col].value_counts().values
fig = px.bar(x=y_col, y=y_values, title=f"{x_col} vs Count")
st.plotly_chart(fig)