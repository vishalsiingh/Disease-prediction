import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import time  # For real-time updates

# Load trained model
model = pickle.load(open("disease_model.pkl", "rb"))

# Load dataset for visualization
df = pd.read_csv("diabetes.csv")  # Ensure dataset is in the same folder

# 🎨 UI Customization
st.set_page_config(page_title="Disease Prediction", page_icon="🩺", layout="wide")

# 🎯 Title & Description
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Disease Prediction System 🏥</h1>", unsafe_allow_html=True)
st.write("### Predict the likelihood of diabetes based on patient health data.")
st.markdown("---")

# 📌 Sidebar for Input
st.sidebar.header("Enter Patient Details 👤")

pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20)
glucose = st.sidebar.number_input("Glucose Level", min_value=0, max_value=200)
bp = st.sidebar.number_input("Blood Pressure", min_value=0, max_value=150)
skin_thickness = st.sidebar.number_input("Skin Thickness", min_value=0, max_value=100)
insulin = st.sidebar.number_input("Insulin Level", min_value=0, max_value=500)
bmi = st.sidebar.number_input("BMI", min_value=0.0, max_value=50.0, step=0.1)
dpf = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, step=0.01)
age = st.sidebar.number_input("Age", min_value=0, max_value=120)

# 🚀 Prediction Button
if st.sidebar.button("Predict 🩺"):
    input_data = np.array([[pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]])
    
    with st.spinner("Analyzing... 🔍"):
        prediction = model.predict(input_data)
    
    st.markdown("---")
    
    if prediction[0] == 1:
        st.markdown("<h2 style='color: red; text-align: center;'>❌ The patient is likely to have diabetes.</h2>", unsafe_allow_html=True)
    else:
        st.markdown("<h2 style='color: green; text-align: center;'>✅ The patient is not likely to have diabetes.</h2>", unsafe_allow_html=True)

st.markdown("---")

# 📊 **Real-Time Data Visualization**
st.subheader("📡 Live Data Tracking & Insights")

# **Session State for Real-Time Updates**
if "update_count" not in st.session_state:
    st.session_state.update_count = 0

# **Simulating Real-Time Updates**
if st.checkbox("🔄 Enable Live Data Tracking", value=True):
    for i in range(10):  # Simulate live data updates for 10 cycles
        df = pd.read_csv("diabetes.csv")  # Reload updated data (simulate live feed)
        
        # 🎯 1. Pie Chart (Diabetes vs. No-Diabetes)
        st.write("### Diabetes Distribution (Live)")
        diabetes_counts = df["Outcome"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(diabetes_counts, labels=["No Diabetes", "Diabetes"], autopct="%1.1f%%", colors=["green", "red"], startangle=90)
        st.pyplot(fig)

        # 🎯 2. Live Line Chart for Glucose Levels
        st.write("### Glucose Trends (Real-Time Data)")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(x=df["Age"], y=df["Glucose"], hue=df["Outcome"], palette="coolwarm", ax=ax)
        st.pyplot(fig)

        # **Auto-Refresh with Session State**
        time.sleep(3)  # Refresh every 3 seconds
        st.session_state.update_count += 1
        st.rerun()  # **New Correct Way to Refresh Streamlit**
