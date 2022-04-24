import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from loader import load_tableau, load_features, load_csv
from utils import predict, get_group_count
from plot import show_probabilities_plot, show_pie_chart, show_multiclass_plot

def view_dashboard():
    st.subheader("Dashboard")
    content_tableau = load_tableau()
    st.components.v1.html(content_tableau, width=1800, height=2200)


def view_form():
    st.subheader("Prediksi Status Kesehatan Keluarga")

    # Load features from json
    features_json = load_features()
    feature_len = len(features_json)

    # Dict to stores user input
    features_user = {}

    # Create features form
    with st.form("Feature"):
        col1, col2 = st.columns([6, 6])
        for index, (feature_name, values) in enumerate(features_json.items()):
            used_col = col1 if index < feature_len // 2 else col2
            with used_col:
                option = st.selectbox(label=feature_name, options=values)
                features_user[feature_name] = option
        is_submit = st.form_submit_button(label="Submit")
    
    if is_submit:
        # Create new dataframe
        user_df = pd.DataFrame(features_user.values(), index=features_user.keys())
        user_df = user_df.T

        # Predicts with user features
        label, prob, classes = predict(user_df)

        # Show Predictions Label
        st.success(f"Terprediksi sebagai **{str(label[0])}**")
        
        # Shot Probability Plot
        fig = show_probabilities_plot(
            labels=classes, 
            values=prob[0], 
            title="Probabilitas Status Kesehatan Keluarga"
        )
        st.pyplot(fig)


def view_eda():
    data = load_csv("sumenep/data/pispk.csv")
    counts = data["Kesehatan Keluarga"].value_counts()
    fig = show_pie_chart(counts)
    st.plotly_chart(fig)

    data = get_group_count(data, ref="MENGGUNAKAN KB", target="Kesehatan Keluarga")

    fig = show_multiclass_plot(data, x="MENGGUNAKAN KB", y="jumlah", hue="Kesehatan Keluarga")
    st.pyplot(fig)
    return