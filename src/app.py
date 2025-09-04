import pandas as pd
import streamlit as st
import numpy as np
import os
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import plotly.express as px
import plotly.graph_objects as go
from predict import predict_spams

# Config de la page
st.set_page_config(
    page_title="Spam Detection App",
    page_icon="🚫📧💣",
    layout="wide",
    initial_sidebar_state="expanded")

# Set up paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "spam.csv"
MODEL_PATH = BASE_DIR / "models" / "model.pkl"

# Load the data
df = pd.read_csv(DATA_PATH, encoding="utf-8")

# App title and description
st.title("📊 Spam Detection App")
st.markdown("""
This application helps you detect spam messages using machine learning.
Upload your own message or try the examples below.
""")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Spam Detection", "Dataset Analysis", "Model Performance"])

with tab1:
    st.header("Spam Message Detection")
    
    # User input for prediction
    user_input = st.text_area("Enter a message to check if it's spam:", 
                             "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005.")
    
    if st.button("Check Message"):
        if user_input:
            # Make prediction
            prediction = predict_spams(user_input)
            
            # Display result with appropriate styling
            if prediction == "spam":
                st.error("🚨 This message is classified as SPAM! 🚨")
            else:
                st.success("✅ This message is classified as HAM (not spam).")
        else:
            st.warning("Please enter a message to analyze.")
    
    # Example messages
    st.subheader("Try these examples:")
    examples = [
        "Congratulations! You've won a $1000 gift card. Click here to claim your prize.",
        "I'm sorry to inform you that your account has been suspended. Please contact support.",
        "Hey, are we still meeting for lunch at noon tomorrow?"
    ]
    
    for i, example in enumerate(examples):
        if st.button(f"Example {i+1}", key=f"example_{i}"):
            prediction = predict_spams(example)
            st.text_area(f"Example {i+1}", example, height=100, key=f"example_text_{i}")
            
            if prediction == "spam":
                st.error("🚨 This message is classified as SPAM! 🚨")
            else:
                st.success("✅ This message is classified as HAM (not spam).")

with tab2:
    st.header("Dataset Analysis")
    
    # Display dataset info
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    
    # Display basic statistics
    st.subheader("Dataset Statistics")
    col1, col2 = st.columns(2)
    
    with col1:
        # Count of spam vs ham
        category_counts = df["Category"].value_counts()
        st.metric("Total Messages", len(df))
        st.metric("Spam Messages", category_counts.get("spam", 0))
        st.metric("Ham Messages", category_counts.get("ham", 0))
    
    with col2:
        # Create a pie chart for spam vs ham distribution
        fig = px.pie(values=category_counts.values, 
                    names=category_counts.index, 
                    title="Distribution of Spam vs Ham Messages",
                    color_discrete_sequence=["#ff9999", "#66b3ff"])
        st.plotly_chart(fig)

with tab3:
    st.header("Model Performance")
    
    # Load the model
    model = joblib.load(MODEL_PATH)
    
    # Split the data (same as in train.py)
    from sklearn.model_selection import train_test_split
    X = df["Message"]
    y = df["Category"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    
    # Get predictions on test set
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label="spam")
    recall = recall_score(y_test, y_pred, pos_label="spam")
    f1 = f1_score(y_test, y_pred, pos_label="spam")
    
    # Display metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Metrics")
        metrics_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
            "Value": [accuracy, precision, recall, f1]
        })
        st.dataframe(metrics_df)
    
    with col2:
        # Create a bar chart for metrics
        fig = px.bar(metrics_df, x="Metric", y="Value", 
                    title="Model Performance Metrics",
                    color="Metric", 
                    color_discrete_sequence=["#ff9999", "#66b3ff", "#99ff99", "#ffcc99"])
        fig.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig)