# Crop-Management

# 🌾 Smart Crop Recommendation System using Machine Learning

A web-based intelligent system built using **Python**, **Machine Learning**, and **Streamlit** to recommend the most appropriate crop for farming based on real-time environmental parameters such as temperature, humidity, pH, and rainfall.

---

## 📌 Objective

Agriculture is the backbone of many economies. However, farmers often lack access to timely and scientific advice on **what to grow** based on environmental factors. This project aims to:

- Help farmers make data-driven decisions.
- Recommend the **most suitable crop** based on real-time parameters.
- Promote **smart farming practices** using Artificial Intelligence.
- Eventually scale into a system that can integrate **fertilizer suggestions**, **weather forecasting**, and **market price tracking**.

---

## 🎯 Project Goal

> **Build a machine learning model that can recommend the most appropriate crop to grow based on key soil and weather parameters, and deploy the model using a clean, interactive Streamlit web interface.**

---

## 🧠 Step-by-Step Development

### 1. 🗂 Dataset Collection & Cleaning
- **Source**: The dataset was taken from publicly available crop recommendation data.
- **Columns**:
  - `temperature`: Temperature in Celsius
  - `humidity`: Relative Humidity in %
  - `ph`: Soil pH value
  - `rainfall`: Rainfall in mm
  - `label`: The target crop (categorical)

### 2. 📊 Exploratory Data Analysis
- Used **Seaborn** and **Matplotlib** to:
  - Visualize crop distribution
  - Understand feature correlation
  - Detect outliers or missing values

### 3. 🔍 Feature Scaling and Label Encoding
- Applied **StandardScaler** to normalize numerical values.
- Encoded crop labels using **LabelEncoder** for model training.

### 4. 🧪 Model Selection and Training
- Tested various algorithms: Decision Tree, Logistic Regression, SVM.
- **Final Choice**: **Random Forest Classifier** due to its high accuracy and robustness.
- Trained on 80% of the dataset, tested on the remaining 20%.

### 5. ✅ Model Evaluation
- Accuracy Score: Over 97% on test data.
- Used classification report and confusion matrix for deeper analysis.

### 6. 💾 Model Saving
- Serialized the model using **joblib/pickle**:
  - `crop_model.pkl` – The trained Random Forest model
  - `scaler.pkl` – Scaler used for input transformation
  - `label_encoder.pkl` – Encoder used to map crops to numerical values

### 7. 🌐 Frontend - Streamlit App
- Built a simple yet clean UI using **Streamlit**.
- Key features:
  - Sidebar image and inspirational quotes
  - Four input sliders for environmental parameters
  - Display of prediction result and corresponding fertilizer suggestion
  - Visualization of dataset on button click

---

## 🖼 Features at a Glance

| Feature                        | Description                                                |
|-------------------------------|------------------------------------------------------------|
| ✅ Crop Recommendation         | Suggests the best crop to grow                             |
| 📈 Dataset Visualization       | Graphs showing data distribution                           |
| 🧪 Scaled Prediction           | Ensures robust model input using standardized values       |
| 🧾 Fertilizer Suggestion       | Displays suggested fertilizers for the recommended crop     |
| 🎨 Clean UI                   | Sidebar image, quotes, and form-based input interface      |

---

## 💻 Run Locally

1. **Clone the repository**:
```bash
git clone https://github.com/your-username/crop-recommendation-app.git
cd crop-recommendation-app
