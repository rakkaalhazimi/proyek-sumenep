from loader import load_label_encoder, load_pipeline

def predict(features):
    label_encoder = load_label_encoder()
    pipeline = load_pipeline()
    prediction = pipeline.predict(features)
    label = label_encoder.inverse_transform(prediction)
    return label
