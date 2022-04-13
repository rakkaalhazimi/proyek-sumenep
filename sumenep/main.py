import streamlit as st
st.set_page_config(layout="wide")

from views import view_dashboard, view_form
from styles import set_style

PAGES = {
    "Dashboard": view_dashboard,
    "Prediksi": view_form,
}

def change_page(page):
    run = PAGES.get(page)
    run()


# Set CSS Custom Style
set_style()

# Set Homepage View
nav_title = st.sidebar.markdown("<p class='title'>Navigasi</p>", unsafe_allow_html=True)
page = st.sidebar.selectbox("", PAGES.keys())
change_page(page)