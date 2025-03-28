import streamlit as st
import sys
import os

# Ensure 'src' and 'app' directories are in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import eda  # Import EDA first
from app import streamlit_app  # Ensure `streamlit_app.py` has a `main()` function
from app import analysis  # Ensure `analysis.py` has a `main()` function

# ✅ Ensure this is the first Streamlit command
st.set_page_config(page_title='Audible Insights', layout='wide')

def main():
    tab1, tab2, tab3 = st.tabs(["Exploratory Data Analysis", "Book Recommendation", "Analysis"])

    with tab1:
        eda.main()  # Ensure `eda.py` has a `main()` function

    with tab2:
        streamlit_app.main()  # Ensure `streamlit_app.py` has a `main()` function

    with tab3:
        analysis.main()  # Ensure `analysis.py` has a `main()` function

if __name__ == "__main__":
    main()
