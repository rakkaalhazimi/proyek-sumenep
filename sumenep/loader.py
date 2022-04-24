import joblib
import json
import os

import streamlit as st
import pandas as pd

import constant as c

@st.cache
def load_features():
    with open(os.path.join(c.BASE_DIR, c.FEATURES_PATH), "r") as file:
        features = json.load(file)
    return features

@st.cache
def load_tableau():
    with open(os.path.join(c.BASE_DIR, c.TABLEAU_PATH), "r") as file:
        content_tableau = file.read()
    return content_tableau

@st.cache
def load_label_encoder():
    fp = os.path.join(c.BASE_DIR, c.LABEL_ENCODER_PATH)
    labenc = joblib.load(fp)
    return labenc

@st.cache
def load_pipeline():
    fp = os.path.join(c.BASE_DIR, c.PIPELINE_PATH)
    pipeline = joblib.load(fp)
    return pipeline

@st.cache
def load_csv(path):
    return pd.read_csv(path)