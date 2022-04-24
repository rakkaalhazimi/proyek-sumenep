from loader import load_label_encoder, load_pipeline

def predict(features):
    label_encoder = load_label_encoder()
    pipeline = load_pipeline()
    
    prediction = pipeline.predict(features)
    probabilities = pipeline.predict_proba(features)
    
    label = label_encoder.inverse_transform(prediction)
    classes = label_encoder.classes_

    return label, probabilities, classes


def get_group_count(df, ref, target):
    return df.groupby([ref, target]).agg(jumlah=(target, "count")).reset_index()
