import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from loader import load_tableau, load_features
from infer import predict

def view_dashboard():
    st.subheader("Dashboard")
    content_tableau = load_tableau()
    st.components.v1.html(content_tableau, width=1800, height=2200)


def view_form():
    st.subheader("Prediksi Status Kesehatan Keluarga")

    features_json = load_features()
    feature_len = len(features_json)
    features_user = {}
    with st.form("Feature"):
        col1, col2 = st.columns([6, 6])

        for index, (feature_name, values) in enumerate(features_json.items()):
            used_col = col1 if index < feature_len // 2 else col2

            with used_col:
                option = st.selectbox(label=feature_name, options=values)
                features_user[feature_name] = option

        is_submit = st.form_submit_button(label="Submit")
    
    if is_submit:
        user_df = pd.DataFrame(features_user.values(), index=features_user.keys())
        user_df = user_df.T

        label, prob, classes = predict(user_df)

        # Show Predictions Label
        st.success(f"Terprediksi sebagai **{str(label[0])}**")
        
        # Shot Probability Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_title("Probabilitas Status Kesehatan Keluarga")
        ax.set_ylim((0, 1.2))
        ax.set_yticks([])

        ## Barplot
        cmap = plt.get_cmap("tab10")
        barplot = ax.bar(x=classes, height=prob[0], width=0.4, color=cmap(prob[0]))

        ## Annotations
        rects = barplot.patches
        fontdict = dict(size=12)
        for rect in rects:
            value = f"{rect.get_height():.2%}"
            ax.text(x=rect.get_x() + 0.1, y=rect.get_height() + 0.05, s=value, fontdict=fontdict)

        
        st.pyplot(fig)