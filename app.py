import streamlit as st
import pandas as pd
import json
import shap
import numpy as np

from streamlit_shap import st_shap
from catboost import CatBoostClassifier

st.set_page_config(layout="wide")

CATBOOST_MODEL_PATH = "models/catboost_fraud_detection_model_resampled.cbm"
TEXT_PATH = "text.json"

model = CatBoostClassifier().load_model(CATBOOST_MODEL_PATH)
explainer = shap.TreeExplainer(model)

with open(TEXT_PATH, "r", encoding="utf-8") as f:
    text = json.load(f)


def predict(data):
    prediction = model.predict(data)
    return prediction[0]


def explain_prediction(input_df):
    shap_values = explainer.shap_values(input_df)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    shap_values = shap_values[0]

    feature_importance = pd.DataFrame({
        "feature": input_df.columns,
        "value": input_df.iloc[0].values,
        "shap_value": shap_values
    }).sort_values(by="shap_value", key=abs, ascending=False)

    explanation = []
    for _, row in feature_importance.head(5).iterrows():
        direction = get_text("feature_increasing") if row["shap_value"] > 0 else get_text("feature_decreasing")
        explanation.append(
            get_text("shap_main_string").format(feature=get_feature_label(row["feature"]),
                                                value=row["value"] if type(row["value"]) == type(np.int64(1)) else
                                                text["features"][row["feature"]]["options"][row["value"]][
                                                    languages[language]],
                                                direction=direction, proba=abs(row["shap_value"]))
        )
    return "\n\n".join(explanation), shap_values


def get_text(key):
    return text[key][languages[language]]


def get_feature_label(key):
    return text["features"][key]["label"][languages[language]]


languages = {"Русский": "ru", "English": "en"}
language = "Русский"
st.sidebar.title(get_text("settings"))
language = st.sidebar.selectbox("Выберите язык / Choose Language", options=list(languages.keys()), index=0)

st.title(get_text("title"))

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    make = st.selectbox(get_feature_label("Make"),
                        options=list(text["features"]["Make"]["options"].keys()),
                        index=0, format_func=lambda x: text["features"]["Make"]["options"][x][languages[language]])
    sex = st.selectbox(get_feature_label("Sex"),
                       options=list(text["features"]["Sex"]["options"].keys()),
                       index=0, format_func=lambda x: text["features"]["Sex"]["options"][x][languages[language]])
    marital_status = st.selectbox(get_feature_label("MaritalStatus"),
                                  options=list(text["features"]["MaritalStatus"]["options"].keys()),
                                  index=0, format_func=lambda x: text["features"]["MaritalStatus"]["options"][x][
            languages[language]])
    age = st.number_input(get_feature_label("Age"), min_value=16, max_value=100)
    fault = st.selectbox(get_feature_label("Fault"),
                         options=list(text["features"]["Fault"]["options"].keys()),
                         index=0, format_func=lambda x: text["features"]["Fault"]["options"][x][languages[language]])

with col2:
    policy_type = st.selectbox(get_feature_label("PolicyType"),
                               options=list(text["features"]["PolicyType"]["options"].keys()),
                               index=0,
                               format_func=lambda x: text["features"]["PolicyType"]["options"][x][languages[language]])
    vehicle_category = st.selectbox(get_feature_label("VehicleCategory"),
                                    options=list(text["features"]["VehicleCategory"]["options"].keys()),
                                    index=0, format_func=lambda x: text["features"]["VehicleCategory"]["options"][x][
            languages[language]])
    vehicle_price = st.number_input(get_feature_label("VehiclePrice"), min_value=1000)
    rep_number = st.number_input(get_feature_label("RepNumber"), min_value=0)
    deductible = st.number_input(get_feature_label("Deductible"), min_value=0)

with col3:
    driver_rating = st.number_input(get_feature_label("DriverRating"), min_value=1, max_value=4)
    days_policy_accident = st.selectbox(get_feature_label("Days:Policy-Accident"),
                                        options=list(text["features"]["Days:Policy-Accident"]["options"].keys()),
                                        index=0,
                                        format_func=lambda x: text["features"]["Days:Policy-Accident"]["options"][x][
                                            languages[language]])
    days_policy_claim = st.selectbox(get_feature_label("Days:Policy-Claim"),
                                     options=list(text["features"]["Days:Policy-Claim"]["options"].keys()),
                                     index=0, format_func=lambda x: text["features"]["Days:Policy-Claim"]["options"][x][
            languages[language]])
    past_number_of_claims = st.selectbox(get_feature_label("PastNumberOfClaims"),
                                         options=list(text["features"]["PastNumberOfClaims"]["options"].keys()),
                                         index=0,
                                         format_func=lambda x: text["features"]["PastNumberOfClaims"]["options"][x][
                                             languages[language]])

with col4:
    age_of_vehicle = st.number_input(get_feature_label("AgeOfVehicle"), min_value=0)
    age_of_policy_holder = st.number_input(get_feature_label("AgeOfPolicyHolder"), min_value=18, max_value=100)
    police_report_filed = st.selectbox(get_feature_label("PoliceReportFiled"),
                                       options=list(text["features"]["PoliceReportFiled"]["options"].keys()),
                                       index=0,
                                       format_func=lambda x: text["features"]["PoliceReportFiled"]["options"][x][
                                           languages[language]])
    witness_present = st.selectbox(get_feature_label("WitnessPresent"),
                                   options=list(text["features"]["WitnessPresent"]["options"].keys()),
                                   index=0, format_func=lambda x: text["features"]["WitnessPresent"]["options"][x][
            languages[language]])

with col5:
    agent_type = st.selectbox(get_feature_label("AgentType"),
                              options=list(text["features"]["AgentType"]["options"].keys()),
                              index=0,
                              format_func=lambda x: text["features"]["AgentType"]["options"][x][languages[language]])
    number_of_suppliments = st.number_input(get_feature_label("NumberOfSuppliments"), min_value=0)
    address_change_claim = st.selectbox(get_feature_label("AddressChange-Claim"),
                                        options=list(text["features"]["AddressChange-Claim"]["options"].keys()),
                                        index=0,
                                        format_func=lambda x: text["features"]["AddressChange-Claim"]["options"][x][
                                            languages[language]])
    number_of_cars = st.number_input(get_feature_label("NumberOfCars"), min_value=1)
    base_policy = st.selectbox(get_feature_label("BasePolicy"),
                               options=list(text["features"]["BasePolicy"]["options"].keys()),
                               index=0,
                               format_func=lambda x: text["features"]["BasePolicy"]["options"][x][languages[language]])

input_data = pd.DataFrame({
    "Make": [make],
    "Sex": [sex],
    "MaritalStatus": [marital_status],
    "Age": [age],
    "Fault": [fault],
    "PolicyType": [policy_type],
    "VehicleCategory": [vehicle_category],
    "VehiclePrice": [vehicle_price],
    "RepNumber": [rep_number],
    "Deductible": [deductible],
    "DriverRating": [driver_rating],
    "Days:Policy-Accident": [days_policy_accident],
    "Days:Policy-Claim": [days_policy_claim],
    "PastNumberOfClaims": [past_number_of_claims],
    "AgeOfVehicle": [age_of_vehicle],
    "AgeOfPolicyHolder": [age_of_policy_holder],
    "PoliceReportFiled": [police_report_filed],
    "WitnessPresent": [witness_present],
    "AgentType": [agent_type],
    "NumberOfSuppliments": [number_of_suppliments],
    "AddressChange-Claim": [address_change_claim],
    "NumberOfCars": [number_of_cars],
    "BasePolicy": [base_policy]
})

if st.button(get_text("predict_button")):
    result = predict(input_data)
    if result == "Yes":
        st.write(get_text("result_fraud"))
    else:
        st.write(get_text("result_no_fraud"))

    explanation_text, shap_values = explain_prediction(input_data)
    st.subheader(get_text("explanation_subheader"))
    st.write(explanation_text)

    st_shap(
        shap.force_plot(
            explainer.expected_value, shap_values, input_data.iloc[0], matplotlib=False
        ),
        height=300
    )
