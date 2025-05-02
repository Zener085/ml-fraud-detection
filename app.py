import streamlit as st
import joblib
import pandas as pd

from catboost import CatBoostClassifier

ENCODER_PATH = "models/onehotencoder.pkl"
SCALER_PATH = "models/scaler.pkl"
CATBOOST_MODEL_PATH = "models/catboost_fraud_detection_model.cbm"

encoder = joblib.load(ENCODER_PATH)
scaler = joblib.load(SCALER_PATH)

model = CatBoostClassifier().load_model(CATBOOST_MODEL_PATH)


def preprocess_data(data: pd.DataFrame):
    categorical_columns = ['Make', 'Sex', 'MaritalStatus', 'Fault', 'PolicyType', 'VehicleCategory',
                           'Days:Policy-Accident', 'Days:Policy-Claim', 'PastNumberOfClaims', 'PoliceReportFiled',
                           'WitnessPresent', 'AgentType', 'AddressChange-Claim', 'BasePolicy']
    data_encoded = encoder.transform(data[categorical_columns])
    encoder_data = pd.DataFrame(
        data_encoded,
        columns=encoder.get_feature_names_out(categorical_columns)
    )
    data = pd.concat([data.drop(categorical_columns, axis=1), encoder_data], axis=1)

    scaled_data = scaler.transform(data)
    return scaled_data


def predict(data):
    processed_data = preprocess_data(data)
    prediction = model.predict(processed_data)
    return prediction[0]


st.title("Предсказание мошенничества")

languages = ["Рус", "En"]
language = st.selectbox("Language/Язык", languages, index=0)

if language == "Рус":
    title_text = "Предсказание мошенничества"
    predict_button_text = "Предсказать"
    result_fraud_text = "Этот человек может быть мошенником."
    result_no_fraud_text = "Этот человек не является мошенником."
else:
    title_text = "Fraud Prediction"
    predict_button_text = "Predict"
    result_fraud_text = "This person may be a fraudster."
    result_no_fraud_text = "This person is not a fraudster."


make = st.selectbox("Make", ["Honda", "Toyota", "Ford", "Mazda", "Chevrolet", "Pontiac", "Acura", "Dodge", "Mercury",
                             "Jaguar", "Nissan", "VW", "Saab", "Saturn", "Porsche", "BMW", "Mercedes", "Ferrari",
                             "Lexus"])
sex = st.selectbox("Sex", ["Male", "Female"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Widow", "Divorced"])
age = st.number_input("Age", min_value=16, max_value=100)
fault = st.selectbox("Fault", ["Policy Holder", "Third Party"])
policy_type = st.selectbox("Policy Type", ["total", "damage", "full"])
vehicle_category = st.selectbox("Vehicle Category", ["Sport", "Utility", "Sedan"])
vehicle_price = st.number_input("Vehicle Price", min_value=1000)
rep_number = st.number_input("Rep Number", min_value=0)
deductible = st.number_input("Deductible", min_value=0)
driver_rating = st.number_input("Driver Rating", min_value=1, max_value=4)
days_policy_accident = st.selectbox("Days left between taking policy and last accident",
                                    ["more than 30", "15 to 30", "none", "1 to 7", "8 to 15"])
days_policy_claim = st.selectbox("Days left between taking policy and claiming the accident",
                                 ["more than 30", "15 to 30", "8 to 15", "none"])
past_number_of_claims = st.selectbox("Past number of claims", ["none", "1", "2 to 4", "more than 4"])
age_of_vehicle = st.number_input("Age Of Vehicle", min_value=0)
age_of_policy_holder = st.number_input("Age Of Policy Holder", min_value=18, max_value=100)
police_report_filed = st.selectbox("Police Report Filed", ["Yes", "No"])
witness_present = st.selectbox("Witness Present", ["Yes", "No"])
agent_type = st.selectbox("Agent Type", ["Internal", "External"])
number_of_suppliments = st.number_input("Number Of Suppliments", min_value=0)
address_change_claim = st.selectbox("Address Change-Claim",
                                    ["1 year", "no change", "4 to 8 years", "2 to 3 years","under 6 months"])
number_of_cars = st.number_input("Number Of Cars", min_value=1)
base_policy = st.selectbox("Base Policy", ["total", "damage", "full"])

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

if st.button("Предсказать"):
    result = predict(input_data)
    if result == "Yes":
        st.write("Этот человек может быть мошенником.")
    else:
        st.write("Этот человек не является мошенником.")
