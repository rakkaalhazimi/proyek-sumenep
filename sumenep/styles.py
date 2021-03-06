import streamlit as st


def set_style():
    st.markdown("""
        <style>
            .title {
                color: white;
                font-size: 48px !important;
            }
            .description {
                font-size: 24px;
            }
            button.step-down, button.step-up {
                color: white;
            }
            button[kind="formSubmit"] {
                background-color: #145EB7;
                color: white;
            }
            button[kind="formSubmit"]:hover {
                background-color: #A0C915;
                color: white;
            }
            div.stRadio > label,
            div.stMultiSelect > label {
                font-size: 24px;
                font-weight: bold;
            }
            div[data-baseweb="select"] > div {
                background-color: white;
            }
            div[data-baseweb="input"] > div > input {
                background-color: white;
            }
            li[role="option"]:hover {
                color: white;
            }
        </style>
        """, unsafe_allow_html=True)


def label_color(label):
    return