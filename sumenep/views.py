import pandas as pd
import streamlit as st
from loader import load_tableau, load_features
from infer import predict

def view_dashboard():
    content_tableau = load_tableau()
    st.components.v1.html(content_tableau, width=1800, height=2200)


def view_form():
    features_json = load_features()

    features_user = {}
    with st.form("Feature"):
        for feature_name, values in features_json.items():
            option = st.selectbox(label=feature_name, options=values)
            features_user[feature_name] = option
        is_submit = st.form_submit_button(label="Submit")
    
    if is_submit:
        st.write("Oke")
        user_df = pd.DataFrame(features_user.values(), index=features_user.keys())
        user_df = user_df.T

        prediction = predict(user_df)
        st.write(prediction)