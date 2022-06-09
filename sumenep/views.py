import pandas as pd
import streamlit as st

from loader import load_tableau, load_features, load_csv, load_pipeline, load_label_encoder
from utils import predict, two_side_selectboxes, two_side_chart, get_group_count
from plot import show_probabilities_plot, show_pie_chart, show_multiclass_plot, show_dt_plot

def view_dashboard():
    st.subheader("Dashboard")
    content_tableau = load_tableau()
    st.components.v1.html(content_tableau, width=1800, height=2200)


def view_prediction():

    # Load feature, model and encoder
    features_json = load_features()
    pipeline = load_pipeline()
    label_encoder = load_label_encoder()

    # Tree Model View
    st.subheader("Diagram Decision Tree")

    ## Get Model
    ohe = pipeline.named_steps["preprocessor"].named_transformers_["onehot"]
    tree_model = pipeline.named_steps["classifier"]

    ## Show Tree
    fig = show_dt_plot(
        model=tree_model, 
        feature_names=ohe.get_feature_names_out(), 
        class_names=label_encoder.classes_
    )
    # fig.savefig("tree.png", dpi=100)
    # st.pyplot(fig)

    # Form View
    st.subheader("Prediksi Status Kesehatan Keluarga")

    

    ## Create features form
    with st.form("Feature"):
        user_inputs = two_side_selectboxes(features_json)
        is_submit = st.form_submit_button(label="Submit")
    
    if is_submit:
        ## Create new dataframe
        user_df = pd.DataFrame(user_inputs.values(), index=user_inputs.keys())
        user_df = user_df.T

        ## Predicts with user features
        label, prob, classes = predict(user_df, pipeline, label_encoder)

        ## Show Predictions Label
        predicted = str(label[0])
        st.success(f"Terprediksi sebagai **{predicted}**")
        
        ## Shot Probability Plot
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

    # Label data
    label = "Kesehatan Keluarga"

    # Count label values
    counts = data[label].value_counts()

    # Label name and order
    label_cats = ["Keluarga Sehat", "Keluarga Pra-Sehat", "Keluarga Tidak Sehat"]
    label_counts = counts.loc[label_cats]
    label_colors = ["green", "yellow", "red"]

    # Show pie chart
    fig = show_pie_chart(values=label_counts, names=label_cats, colors=label_colors)
    st.plotly_chart(fig)

    # Create data for bar charts
    unused_col = ["Kesehatan Keluarga", "Kesehatan Keluarga Label", "IKS INTI", "NAMA KK"]
    viz_data = data.drop(columns=unused_col)

    # Create bar charts
    figs = []
    count_name = "jumlah"
    for x in viz_data.columns:
        group_count_df = get_group_count(data, ref=x, target=label, name=count_name)
        fig = show_multiclass_plot(df=group_count_df, x=x, y=count_name, hue=label, colors=label_colors, order=label_cats)
        figs.append(fig)
    
    # Show multiple bar charts
    two_side_chart(figs)