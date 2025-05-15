# Fraud Detection in Auto Insurance Claims
This project develops a machine learning solution to detect fraudulent auto insurance claims, addressing a critical issue that leads to significant financial losses for insurance companies and higher premiums for policyholders. By accurately identifying fraudulent claims, this work aims to enhance efficiency and fairness in insurance processing.

The main objectives are:

- To build a robust machine learning model, leveraging CatBoost, to distinguish between legitimate and fraudulent claims.
- To provide an interpretable web interface using Streamlit, where users can input claim details and receive predictions with SHAP-based explanations.

This project was conducted as part of a thesis titled "Improving Machine Learning Models for Fraud Classification in Car Insurance" at university.

## Repository Structure
- `notebooks/`: Contains Jupyter notebooks for data preprocessing and model training.
  - `fraud_detection_training.ipynb`: Notebook for training the CatBoost model.
- `app.py`: Streamlit web application for interactive fraud prediction.
- `models/`: Directory containing the trained CatBoost model (`catboost_fraud_detection_model_resampled.cbm`).
- `requirements.txt`: List of Python dependencies required to run the project.

## Dataset
The dataset used is the "Vehicle Insurance Fraud Detection" dataset, sourced from Kaggle by user "khusheekapoor" (Vehicle Insurance Fraud Detection). It includes features such as vehicle make, policyholder age, and claim details, with a target variable indicating fraud. To retrain the model, download the dataset and place it in a `data/` directory.

## Installation and Setup
To run the project, you need Python 3.12, Jupyter Notebook, and the following libraries: CatBoost, SHAP, Streamlit, and others listed in `requirements.txt`.

1. Clone the repository:
   ```bash
   git clone https://github.com/Zener085/ml-fraud-detection.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

The app uses a pre-trained CatBoost model in the `models/` directory. To retrain the model:

1. Download the dataset from Kaggle.
2. Place it in the `data/` directory.
3. Execute the `fraud_detection_training.ipynb` notebook.

## Web Application
The Streamlit web application offers an interactive interface for predicting whether an auto insurance claim is fraudulent. Users can input details like vehicle make, policyholder sex, age, and other claim-related features. The app supports bilingual functionality in English and Russian, enhancing accessibility.

### Features
- **Multilingual Support**: Switch between English and Russian interfaces.
- **Input Fields**: Enter claim details via user-friendly forms.
- **Fraud Prediction**: Receive a probability score indicating the likelihood of fraud, powered by a pre-trained CatBoost model.
- **SHAP Explanations**: View real-time SHAP (SHapley Additive exPlanations) plots showing how each feature contributes to the prediction.

SHAP is a game-theoretic approach that explains model predictions by assigning importance values to features. In this app, SHAP values highlight factors driving fraud predictions, such as high vehicle price or recent address changes. This transparency is vital for fraud detection, ensuring decisions are fair, understandable, and trustworthy.

## License

This project is licensed under the MIT License. See the [LICENSE file](https://github.com/Zener085/ml-fraud-detection/blob/main/LICENSE) for details.
