import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Load trained model and preprocessors
model = pickle.load(open("crop_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# Load dataset for visualization
df = pd.read_csv("Crop_recommendation.csv")

# Define fertilizer recommendations
fertilizer_suggestions = {
    "rice": "Urea, DAP, Potash",
    "wheat": "NPK 10-26-26, Zinc Sulphate",
    "maize": "DAP, Urea, Zinc",
    "cotton": "Nitrogen, Potassium, Phosphorus",
    "banana": "DAP, MOP, Zn",
    "mango": "NPK 10-26-26, Organic Manure",
}

# Agriculture slogans with authors
slogans = [
    ("🌱 Agriculture is the most healthful, most useful, and most noble employment of man.", "— George Washington"),
    ("🌾 The discovery of agriculture was the first big step toward a civilized life.", "— Arthur Keith"),
    ("🌍 Agriculture not only gives riches to a nation, but the only riches she can call her own.", "— Samuel Johnson"),
    ("🌞 The ultimate goal of farming is not the growing of crops, but the cultivation of human beings.", "— Masanobu Fukuoka"),
    ("🌿 Farming looks mighty easy when your plow is a pencil and you're a thousand miles from the cornfield.", "— Dwight D. Eisenhower"),
]

# List of random agriculture-related images
image_urls = [
    "2.jpeg",
]

# Randomly select a slogan and an image
quote, author = random.choice(slogans)
image_url = random.choice(image_urls)

# Page Configuration
st.set_page_config(page_title="🌱 Smart Crop Advisor", layout="wide")

# Sidebar with styling
st.sidebar.markdown(
    """
    <style>
        .sidebar-text { font-size: 16px; font-weight: bold; color: #4CAF50; }
        .sidebar-title { font-size: 22px; font-weight: bold; color: #FF5722; text-align: center; }
        .sidebar-img { text-align: center; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown('<p class="sidebar-title">🌾 Smart Crop Advisor</p>', unsafe_allow_html=True)
st.sidebar.markdown('<p class="sidebar-text">👋 Welcome! Choose a section below:</p>', unsafe_allow_html=True)

# Image section in sidebar (Updated use_column_width → use_container_width)
st.sidebar.markdown('<div class="sidebar-img">', unsafe_allow_html=True)
st.sidebar.image(image_url, caption="🌿 Agriculture & Farming", use_container_width=True)
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Sidebar Navigation
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Data Insights", "🌾 Predict Crop"])

# Random Agriculture Quote
st.sidebar.markdown(f"🌟 *{quote}*  \n**{author}**")

# Home Page
if page == "🏠 Home":
    st.title("🌱 Welcome to Smart Crop Advisor")
    st.write(
        """
        Agriculture is the backbone of any nation. This system helps farmers and researchers **choose the best crop** 
        based on **soil and climate conditions** using **Machine Learning**.  
        
        🚀 **How it Works?**  
        1. Enter **soil and weather parameters** (Nitrogen, Temperature, pH, Rainfall, etc.).  
        2. Our trained **Machine Learning model** analyzes the data.  
        3. It suggests the **most suitable crop** for your land.  
        4. You also get **fertilizer recommendations** for better productivity.  
        """
    )
    st.image("1.jpeg", use_container_width=True)
    st.success("👈 Explore Data Insights or Predict a Crop using the sidebar!")

# Data Insights Page
elif page == "📊 Data Insights":
    st.title("📊 Crop Data Insights & Trends")

    with st.expander("📋 View Dataset (First 5 Rows)"):
        st.dataframe(df.head())

    st.subheader("🌾 Crop Distribution Across Dataset")
    fig, ax = plt.subplots()
    sns.countplot(y=df["label"], order=df["label"].value_counts().index, palette="viridis", ax=ax)
    ax.set_title("Distribution of Different Crops")
    st.pyplot(fig)
    st.info("📌 This chart shows the **frequency of each crop** in our dataset. The most frequent crops are commonly grown.")

    st.subheader("🌡 Temperature vs. Crop Growth")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(x="label", y="temperature", data=df, palette="coolwarm", ax=ax)
    plt.xticks(rotation=90)
    st.pyplot(fig)
    st.info("📌 This graph shows the **temperature range** for different crops. You can see that some crops thrive in warm conditions while others prefer cooler temperatures.")

# Crop Prediction Page
elif page == "🌾 Predict Crop":
    st.title("🌾 Crop Prediction & Fertilizer Advice")
    st.write("Enter the environmental conditions below:")

    col1, col2 = st.columns(2)

    with col1:
        N = st.number_input("🌿 Nitrogen (N)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
        P = st.number_input("🌾 Phosphorus (P)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
        K = st.number_input("🪴 Potassium (K)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)

    with col2:
        temp = st.number_input("🌡 Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0, step=0.1)
        humidity = st.number_input("💧 Humidity (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
        ph = st.number_input("🧪 pH Level", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
        rainfall = st.number_input("🌧 Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0, step=0.1)

    if st.button("🚀 Predict Crop"):
        input_data = np.array([[N, P, K, temp, humidity, ph, rainfall]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        predicted_crop = label_encoder.inverse_transform(prediction)[0]

        st.success(f"🌾 Recommended Crop: **{predicted_crop}**")
        st.info(f"🌿 Suggested Fertilizer: **{fertilizer_suggestions.get(predicted_crop, 'General NPK Fertilizer')}**")
