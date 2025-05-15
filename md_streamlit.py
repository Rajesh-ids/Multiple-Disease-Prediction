import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from streamlit_option_menu import option_menu
from sklearn.preprocessing import StandardScaler


# Load model
with open('D:\project\MultipleDiseasePrediction\lp.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Set page configuration
st.set_page_config(page_title="Employee Portal & EDA", layout="wide")


# Sidebar Navigation Menu
with st.sidebar:
    selected = option_menu(
        menu_title="Health Prediction", 
        options=[
            "Liver Patient Prediction",
            "Kidney Disease Prediction",
            "Parkinsons Prediction"
        ],
        icons=["droplet-half", "heart-pulse", "activity"],
        menu_icon="cast",
        default_index=0
    )

# <-- liver patient prediction -->

# Define the function for prediction
def predict_status(features_1):
    # Feature scaling
    scaler = StandardScaler()
    features_1_scaled = scaler.fit_transform([features_1])
    prediction = loaded_model.predict(features_1_scaled)
    return prediction[0]

# Liver Prediction Form
if selected == "Liver Patient Prediction":
    st.title("Liver Patient Prediction")

    st.write("Enter the following health metrics:")

    # Input fields for features (X)
    age = st.number_input("Age", min_value=4, max_value=90, step=1)
    total_bilirubin = st.number_input("Total Bilirubin", min_value=0.4, step=42.8)
    alk_phosphotase = st.number_input("Alkaline Phosphotase", min_value=63, step=2110)
    alt = st.number_input("Alamine Aminotransferase (ALT)", min_value=10, step=2000)
    ast = st.number_input("Aspartate Aminotransferase (AST)", min_value=10, step=4929)
    total_proteins = st.number_input("Total Proteins", min_value=2.7, step=9.6)
    albumin = st.number_input("Albumin", min_value=0.9, step=5.5)

    # Collect features
    features_1 = [age,  total_bilirubin, alk_phosphotase, alt, ast,total_proteins,albumin]

    # Button for prediction
    if st.button('Get Result'):
        result = predict_status(features_1)
        
        if result == 1:
            st.success('The status is: 1 (Liver Disease)')
        else:
            st.success('The status is: 2 (No Liver Disease)')



#<-- Parkinsons Prediction -->
# test1
# Load the saved model
with open('pp.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the function for prediction
def predict_status(features_2):
    # Feature scaling
    scaler = StandardScaler()
    features_2_scaled = scaler.fit_transform([features_2])
    prediction = model.predict(features_2_scaled)
    return prediction[0]

if selected == "Parkinsons Prediction":
    # Streamlit UI
    st.title('Parkinson\'s Disease Prediction')
    st.write('Enter the values for the features and click "Get Result" to predict the status.')

    # Input fields for the features
    mdvp_fo_hz = st.number_input('MDVP_Fo_Hz', min_value=0.0, value=0.0)
    mdvp_fhi_hz = st.number_input('MDVP_Fhi_Hz', min_value=0.0, value=0.0)
    mdvp_jitter_percent = st.number_input('MDVP_Jitter (%)', min_value=0.0, value=0.0)
    mdvp_jitter_abs = st.number_input('MDVP_Jitter Abs', min_value=0.0, value=0.0)
    mdvp_rap = st.number_input('MDVP_Rap', min_value=0.0, value=0.0)
    mdvp_ppq = st.number_input('MDVP_Ppq', min_value=0.0, value=0.0)
    jitter_ddp = st.number_input('Jitter DDP', min_value=0.0, value=0.0)
    shimmer_apq3 = st.number_input('Shimmer APQ3', min_value=0.0, value=0.0)
    shimmer_apq5 = st.number_input('Shimmer APQ5', min_value=0.0, value=0.0)
    mdvp_apq = st.number_input('MDVP_APQ', min_value=0.0, value=0.0)
    shimmer_dda = st.number_input('Shimmer DDA', min_value=0.0, value=0.0)
    nhr = st.number_input('NHR', min_value=0.0, value=0.0)
    rpde = st.number_input('RPDE', min_value=0.0, value=0.0)
    dfa = st.number_input('DFA', min_value=0.0, value=0.0)
    spread1 = st.number_input('Spread1', min_value=0.0, value=0.0)
    spread2 = st.number_input('Spread2', min_value=0.0, value=0.0)
    d2 = st.number_input('D2', min_value=0.0, value=0.0)
    ppe = st.number_input('PPE', min_value=0.0, value=0.0)

    # Collect features
    features_2 = [
        mdvp_fo_hz, mdvp_fhi_hz, mdvp_jitter_percent, mdvp_jitter_abs, mdvp_rap,
        mdvp_ppq, jitter_ddp, shimmer_apq3, shimmer_apq5, mdvp_apq, shimmer_dda,
        nhr, rpde, dfa, spread1, spread2, d2, ppe
    ]

    # Button for prediction
    if st.button('Get Result'):
        result = predict_status(features_2)
        
        if result == 1:
            st.success('The status is: 1 (Parkinson\'s Disease)')
        else:
            st.success('The status is: 0 (No Parkinson\'s Disease)')


# <-- Kidney Disease Prediction -->
# test2
# Load the trained model
with open('kp.pkl', 'rb') as file:
    kidney_model = pickle.load(file)

# Define the function for prediction
def predict_status(features_3):
    # Feature scaling
    scaler = StandardScaler()
    features_3_scaled = scaler.fit_transform([features_3])
    prediction = kidney_model.predict(features_3_scaled)
    return prediction[0]


if selected == "Kidney Disease Prediction":
    # Title
    st.title("Kidney Disease Prediction App")

    # Create input fields for each feature
    age = st.number_input("Age", min_value=0)
    bp = st.number_input("Blood Pressure", min_value=0)
    sg = st.number_input("Specific Gravity", min_value=1.000, max_value=1.030, step=0.001, format="%.3f")
    al = st.number_input("Albumin", min_value=0, max_value= 5)
    pc = st.selectbox("Pus Cell", options=[0, 1], format_func=lambda x: "Normal" if x == 0 else "Abnormal")
    pcc = st.selectbox("Pus Cell Clumps", options=[0, 1], format_func=lambda x: "Not Present" if x == 0 else "Present")
    ba = st.selectbox("Bacteria", options=[0, 1], format_func=lambda x: "Not Present" if x == 0 else "Present")
    sc = st.number_input("Serum Creatinine", min_value=0.0, max_value = 50.0)
    hemo = st.number_input("Hemoglobin", min_value=0.0, max_value = 18.0)
    htn = st.selectbox("Hypertension", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    dm = st.selectbox("Diabetes Mellitus", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    cad = st.selectbox("Coronary Artery Disease", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    appet = st.selectbox("Appetite", options=[0, 1], format_func=lambda x: "Poor" if x == 0 else "Good")
    pe = st.selectbox("Pedal Edema", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    ane = st.selectbox("Anemia", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")


   # Collect features
    features_3 = [age, bp, sg, al, pc, pcc, ba, sc, hemo, htn, dm, cad, appet, pe, ane]

    # Button for prediction
    if st.button('Get Result'):
        result = predict_status(features_3)
        
        if result == 1:
            st.success('The status is: 1 (Kidney Disease)')
        else:
            st.success('The status is: 0 (No Kidney Disease)')



