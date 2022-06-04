import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from loader import load_tableau, load_features, load_csv
from utils import predict, two_side_selectboxes, two_side_chart
from plot import show_probabilities_plot, show_pie_chart, make_figure

def view_dashboard():
    st.subheader("Dashboard")
    content_tableau = load_tableau()
    st.components.v1.html(content_tableau, width=1800, height=2200)


def view_form():
    st.subheader("Prediksi Status Kesehatan Keluarga")

    # Load features from json
    features_json = load_features()

    # Create features form
    with st.form("Feature"):
        user_inputs = two_side_selectboxes(features_json)
        is_submit = st.form_submit_button(label="Submit")
    
    if is_submit:
        # Create new dataframe
        user_df = pd.DataFrame(user_inputs.values(), index=user_inputs.keys())
        user_df = user_df.T

        # Predicts with user features
        label, prob, classes = predict(user_df)

        # Show Predictions Label
        predicted = str(label[0])
        st.success(f"Terprediksi sebagai **{predicted}**")
        
        # Shot Probability Plot
        fig = show_probabilities_plot(
            labels=classes, 
            values=prob[0], 
            title="Probabilitas Status Kesehatan Keluarga"
        )
        st.pyplot(fig)


def view_eda():
    st.subheader("Exploratory Data Analysis")

    # Load data
    data = load_csv("sumenep/data/pispk.csv")

    # Show pie chart
    counts = data["Kesehatan Keluarga"].value_counts()
    fig = show_pie_chart(counts)
    st.plotly_chart(fig)

    # Show bar chart
    label_order = ["Keluarga Sehat", "Keluarga Pra-Sehat", "Keluarga Tidak Sehat"]
    viz_data = data.drop(columns=["Kesehatan Keluarga", "Kesehatan Keluarga Label", "IKS INTI", "NAMA KK"])
    figs = make_figure(data, X=viz_data.columns, y="jumlah", hue="Kesehatan Keluarga", order=label_order)
    two_side_chart(figs)