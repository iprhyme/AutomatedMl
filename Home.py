import streamlit as st
from st_pages import Page, show_pages
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

st.title("Automated Machine Learning and Data Science")
st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")

with st.expander("What does this website do?"):
    st.write("this website automates some of the necessary tasks of Machine Learning and Data Science.")
    st.write("from EDA, to Preprocessing, to Visualization of the data, to Building the Model")
    st.write("you only need to upload your dataset! and check a few things about your dataset only!")

st.subheader("So, What Are You Waiting For?!")
st.text("head to the Upload Section and begin your journey!")

with st.expander("Contacts Information"):
    st.text("Contact Us via:")
    st.markdown("[Twitter](https://x.com/_YazeedA)")
    st.markdown("[LinkedIn](https://www.linkedin.com/in/yazeed-alobaidan-218b4a2b4/)")
    st.markdown("[GitHub](https://github.com/iprhyme)")
    




