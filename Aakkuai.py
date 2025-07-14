import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from transformers import pipeline
from langchain_together import ChatTogether
import os

# Set API key
os.environ["TOGETHER_API_KEY"] = "tgp_v1_4hJBRX0XDlwnw_hhUnhP0e_lpI-u92Xhnqny2QIDAIM"

# Initialize LLM with Together AI endpoint
llm = ChatTogether(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    temperature=0,
    max_tokens=512,
)

# Streamlit app
st.title("Aakku AI - Machine Learning and Deep Learning App")

# Create two columns
col1, col2 = st.columns([2, 3])

with col1:
    st.header("Features")
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Data Exploration")
        st.write(data.head())
        st.write(data.info())
        st.write(data.describe())

        # Basic EDA
        st.write("Basic EDA")
        st.write("Data Shape:", data.shape)
        st.write("You have", len(data.columns), "features and", data.shape[0], "rows")

        # Machine learning features
        st.write("Machine Learning Features")
        ml_features = ["Linear Regression", "Logistic Regression", "Decision Tree Regressor", "Decision Tree Classifier", "Random Forest Regressor", "Random Forest Classifier"]
        selected_ml_feature = st.selectbox("Select a machine learning feature", ml_features)

        # NLP features
        st.write("NLP Features")
        nlp_features = ["Sentiment Analysis", "Text Classification"]
        selected_nlp_feature = st.selectbox("Select an NLP feature", nlp_features)

with col2:
    st.header("Aakku AI")
    user_query = st.text_input("Enter your query")
    if user_query:
        response = llm.invoke(user_query)
        st.write("Aakku AI:", response)

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        data = data.dropna()  # drop rows with missing values
        X = data.drop('target', axis=1)
        y = data['target']

        if selected_ml_feature:
            # Perform machine learning task
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            if selected_ml_feature == "Linear Regression":
                model = LinearRegression()
            elif selected_ml_feature == "Logistic Regression":
                model = LogisticRegression()
            elif selected_ml_feature == "Decision Tree Regressor":
                model = DecisionTreeRegressor()
            elif selected_ml_feature == "Decision Tree Classifier":
                model = DecisionTreeClassifier()
            elif selected_ml_feature == "Random Forest Regressor":
                model = RandomForestRegressor()
            elif selected_ml_feature == "Random Forest Classifier":
                model = RandomForestClassifier()

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.write("Model Performance:", accuracy_score(y_test, y_pred))

        if selected_nlp_feature:
            # Perform NLP task
            text = data['text']
            if selected_nlp_feature == "Sentiment Analysis":
                sentiment_pipeline = pipeline('sentiment-analysis')
                sentiment = sentiment_pipeline(text)
                st.write("Sentiment:", sentiment)
