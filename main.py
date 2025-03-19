import streamlit as st
import pandas as pd
import numpy as np
import pickle
from catboost import CatBoostClassifier

st.set_page_config(
    page_title="Fraud Detection",
    page_icon="🚗",
    layout="wide"
)


# st.title("")

@st.cache_resource
def load_model():
    _model = CatBoostClassifier()
    _model.load_model("models/catboost_fraud_detection_model.cbm")
    return _model


@st.cache_data
def load_feature_names():
    with open("models/feature_names.pkl", "rb") as f:
        _feature_names = pickle.load(f)
    return _feature_names


model = load_model()
feature_names = load_feature_names()


def create_input_fields():
    global feature_names
    inputs = {}

    st.sidebar.header("Enter your data")
    st.sidebar.markdown(
        "To calculate a probability to check the car or the potential policy holder, please enter their"
        " data"
    )

    client_features = [f for f in feature_names if f.lower() in ("age", "agenttype", "sex", "maritalstatus")]
    vehicle_features = [f for f in feature_names if "vehicle" in f.lower() and f != "NumberOfCars_2 vehicles"]
    policy_features = [f for f in feature_names if
                       ("policy" in f.lower() or "car" in f.lower()) and "accident" not in f.lower() and "claim" not in f.lower()]
    claim_features = [f for f in feature_names if "claim" in f.lower() or "accident" in f.lower()]
    other_features = [f for f in feature_names if
                      f not in client_features + vehicle_features + policy_features + claim_features]


    def create_group_inputs(features, group_name):
        if features:
            st.sidebar.subheader(f"{group_name}")
            for feature in features:
                inputs[feature] = st.sidebar.number_input(f"{feature}", value=0.0, step=0.1)

    create_group_inputs(client_features, "Client data")
    create_group_inputs(vehicle_features, "Vehicle data")
    create_group_inputs(policy_features, "Policy data")
    create_group_inputs(claim_features, "Claims data")
    create_group_inputs(other_features, "Other info")

    return inputs


inputs = create_input_fields()

if st.sidebar.button("Check"):
    input_data = pd.DataFrame([inputs], columns=feature_names)

    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction_class = model.predict(input_data)[0]

    st.header("The result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="the probability to get frauded",
            value=f"{prediction_proba:.2%}"
        )

        if prediction_proba < 0.3:
            risk_level = "Low"
            color = "green"
        elif prediction_proba < 0.7:
            risk_level = "Medium"
            color = "orange"
        else:
            risk_level = "High"
            color = "red"

        st.markdown(f"<h3 style='color: {color}'>Level of risk: {risk_level}</h3>", unsafe_allow_html=True)
else:
    st.info("Please enter the info about you and your car on the left.")
