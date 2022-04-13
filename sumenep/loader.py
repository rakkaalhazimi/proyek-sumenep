import joblib
import json
import os
import constant as c

def load_features():
    with open(os.path.join(c.BASE_DIR, c.FEATURES_PATH), "r") as file:
        features = json.load(file)
    return features


def load_tableau():
    with open(os.path.join(c.BASE_DIR, c.TABLEAU_PATH), "r") as file:
        content_tableau = file.read()
    return content_tableau


def load_label_encoder():
    fp = os.path.join(c.BASE_DIR, c.LABEL_ENCODER_PATH)
    labenc = joblib.load(fp)
    return labenc


def load_pipeline():
    fp = os.path.join(c.BASE_DIR, c.PIPELINE_PATH)
    pipeline = joblib.load(fp)
    return pipeline