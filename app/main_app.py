import streamlit as st
import sys
import os

# Ensure 'src' and 'app' directories are in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import eda  # Import EDA first

def main():
    st.set_page_config(page_title='Data Analysis & Visualization', layout='wide')

    tab1, tab2 = st.tabs(["Exploratory Data Analysis", "Streamlit App"])

    with tab1:
        eda.main()  # Ensure `eda.py` has a `main()` function

    with tab2:
        from app import streamlit_app  # Now `app` is a package
        streamlit_app.main()  # Ensure `streamlit_app.py` has a `main()` function

if __name__ == "__main__":
    main()
