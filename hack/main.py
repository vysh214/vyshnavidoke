import streamlit as st
from symptoms_detection import app as symptoms_detection_app
from doctor_handwriting import app as doctor_handwriting_app
from disease_dect import app as disease_dect_app

# Sidebar for navigation
st.sidebar.title("Navigation")
app_selection = st.sidebar.radio("Select an App", ("Symptoms Detection", "Doctor Handwriting", "Disease Detection"))

# Load selected app
if app_selection == "Symptoms Detection":
    symptoms_detection_app()
elif app_selection == "Doctor Handwriting":
    doctor_handwriting_app()
elif app_selection == "Disease Detection":
    disease_dect_app()
