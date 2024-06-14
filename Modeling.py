from sklearn import svm
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    classification_report,
    accuracy_score,
    roc_curve,
    auc,
    confusion_matrix,
)

import pickle
import io
import os
from st_pages import Page, show_pages, add_page_title

# Modeling section
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

st.title("LET'S BUILD THE MACHINE LEARNING MODEL!!")
sc_choice = st.radio(
    "To standardize the data, should we use StandardScaler or MinMaxScaler?",
    ["leave it as it is", "StandardScaler", "MinMaxScaler"],
)

if sc_choice == "leave it as it is":
    df_scaled = df
else:
    if st.session_state.classification:
        scaler = (
            StandardScaler() if sc_choice == "StandardScaler" else MinMaxScaler()
        )
        X_scaled = scaler.fit_transform(
            df.drop(st.session_state.chosen_target, axis=1)
        )
        df_scaled = pd.DataFrame(
            X_scaled,
            columns=df.drop(st.session_state.chosen_target, axis=1).columns,
        )
        df_scaled[st.session_state.chosen_target] = df[
            st.session_state.chosen_target
        ]
    else:
        scaler = (
            StandardScaler() if sc_choice == "StandardScaler" else MinMaxScaler()
        )
        df_scaled = scaler.fit_transform(df)
        df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
st.dataframe(df_scaled)

test_split = st.slider(
    "Choose the test split", min_value=0.1, max_value=1.00, value=0.2
)
x = df_scaled.drop(st.session_state.chosen_target, axis=1)
y = df_scaled[st.session_state.chosen_target]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=test_split, random_state=42
)

if st.session_state.classification:
    models = {
        "Logistic Regression": LogisticRegression(),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
        "Random Forest": RandomForestClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "SVM": svm.SVC(kernel="linear"),
        "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
        "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis(),
    }
    model_choice = st.selectbox("Select a model", list(models.keys()))
    st.session_state.chosen_model = models[model_choice]
else:
    models = {
        "Linear Regression": LinearRegression(),
        "SVM": svm.SVR(),
        "Random Forest": RandomForestRegressor(),
    }
    model_choice = st.selectbox("Select a model", list(models.keys()))
    st.session_state.chosen_model = models[model_choice]

chosen_model = st.session_state.chosen_model

chosen_model.fit(x_train, y_train)
y_pred = chosen_model.predict(x_test)
if st.button("Predict using " + str(chosen_model)):
    if st.session_state.classification:
        accuracy = accuracy_score(y_test, y_pred)
        st.write("The accuracy is: " + str(accuracy))
        st.write("Classification Report:\n", classification_report(y_test, y_pred))
    else:
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        st.write("R2 score is: " + str(r2))
        st.write("MSE is: " + str(mse))

    if st.session_state.classification:

        st.write("confusion matrix:")
        st.write(confusion_matrix(y_test, y_pred))

        st.subheader("ROC Curve for the Models")
        plt.figure(figsize=(10, 8))

        fpr, tpr, _ = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)

        fig2 = px.line(
            x=fpr,
            y=tpr,
            title=f"ROC Curve for the {chosen_model} (AUC = {roc_auc:.2f})",
            labels={"x": "False Positive Rate", "y": "True Positive Rate"},
        )

        st.plotly_chart(fig2)
    model_file = io.BytesIO()
    pickle.dump(chosen_model, model_file)
    model_file.seek(0)

    # Provide a download button
    st.download_button(
        label="Download the Model!",
        data=model_file,
        file_name="model.pkl",
        mime="application/octet-stream",
    )