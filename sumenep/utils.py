import streamlit as st

from loader import load_label_encoder, load_pipeline


def predict(features):
    label_encoder = load_label_encoder()
    pipeline = load_pipeline()
    
    prediction = pipeline.predict(features)
    probabilities = pipeline.predict_proba(features)
    
    label = label_encoder.inverse_transform(prediction)
    classes = label_encoder.classes_

    return label, probabilities, classes

@st.cache
def get_group_count(df, ref, target):
    return df.groupby([ref, target]).agg(jumlah=(target, "count")).reset_index()


def two_side_selectboxes(user_items):
    # Dict to stores user input
    user_inputs = {}
    input_len = len(user_items)

    col1, col2 = st.columns([6, 6])
    for index, (feature_name, values) in enumerate(user_items.items()):
        used_col = col1 if index < input_len // 2 else col2
        with used_col:
            option = st.selectbox(label=feature_name, options=values)
            user_inputs[feature_name] = option

    return user_inputs


def two_side_chart(figures):
    col1, col2 = st.columns([6, 6])
    fig_len = len(figures)
    for index, fig in enumerate(figures):
        used_col = col1 if index < fig_len // 2 else col2
        with used_col:
            st.pyplot(fig)