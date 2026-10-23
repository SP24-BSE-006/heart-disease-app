import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

MODEL_PATH = Path(__file__).parent / "svc_trained_model.pkl"


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def build_label_encoders():
    encoders = {}
    encoders["Gender"] = LabelEncoder().fit(["Male", "Female"])
    encoders["ChestPain"] = LabelEncoder().fit(
        ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"]
    )
    return encoders


model = load_model()
encoders = build_label_encoders()

st.title("❤️ Heart Disease Predictor")
st.markdown(
    "Enter a patient's details and the trained Support Vector Classifier will "
    "predict whether they are likely to have heart disease. "
    "Built from the UCI Heart Disease (Cleveland) dataset using 4 input attributes."
)

st.divider()

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=55, step=1)
    gender = st.selectbox("Gender", ["Male", "Female"])
with col2:
    chestpain = st.selectbox(
        "Chest Pain Type",
        ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"],
    )
    maxhr = st.number_input("Max Heart Rate (bpm)", min_value=40, max_value=250, value=150, step=1)

if st.button("Predict", type="primary", use_container_width=True):
    user_input = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "ChestPain": [chestpain],
        "MaxHeartRate": [maxhr],
    })

    encoded = pd.DataFrame({
        "Age": user_input["Age"],
        "Gender": encoders["Gender"].transform(user_input["Gender"]),
        "ChestPain": encoders["ChestPain"].transform(user_input["ChestPain"]),
        "MaxHeartRate": user_input["MaxHeartRate"],
    })

    prediction = model.predict(encoded)[0]

    st.divider()
    if prediction == 1:
        st.error("### ⚠️ Heart disease likely")
    else:
        st.success("### ✅ Heart disease not likely")

    with st.expander("See the encoded feature vector sent to the model"):
        st.dataframe(encoded, hide_index=True)

st.divider()
st.caption(
    "Learning project built for a university assignment. It is not a medical tool "
    "and must not be used for real diagnosis."
)
