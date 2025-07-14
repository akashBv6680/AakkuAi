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

# Function to load data
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")

# Function to perform data exploration
def explore_data(data):
    st.write("Data Exploration")
    st.write(data.head())
    st.write(data.info())
    st.write(data.describe())

    # Basic EDA
    st.write("Basic EDA")
    st.write("Data Shape:", data.shape)
    st.write("You have", len(data.columns), "features and", data.shape[0], "rows")

    # Plot histograms for numerical features
    numerical_features = data.select_dtypes(include=['int64', 'float64']).columns
    fig, axes = plt.subplots(nrows=len(numerical_features), ncols=1, figsize=(8, 6*len(numerical_features)))
    for i, feature in enumerate(numerical_features):
        sns.histplot(data[feature], ax=axes[i])
        axes[i].set_title(f"Histogram of {feature}")
    st.pyplot(fig)

    # Plot bar charts for categorical features
    categorical_features = data.select_dtypes(include=['object']).columns
    fig, axes = plt.subplots(nrows=len(categorical_features), ncols=1, figsize=(8, 6*len(categorical_features)))
    for i, feature in enumerate(categorical_features):
        sns.countplot(data[feature], ax=axes[i])
        axes[i].set_title(f"Bar chart of {feature}")
    st.pyplot(fig)

# Function to preprocess data
def preprocess_data(data):
    categorical_features = data.select_dtypes(include=['object']).columns
    for feature in categorical_features:
        st.write(f"Feature: {feature}")
        encoding_type = st.selectbox(f"Select encoding type for {feature}", ["Label Encoding", "One-Hot Encoding"])
        if encoding_type == "Label Encoding":
            le = LabelEncoder()
            data[feature] = le.fit_transform(data[feature])
        elif encoding_type == "One-Hot Encoding":
            data = pd.get_dummies(data, columns=[feature])
    return data

# Function to build machine learning models
def build_ml_models(X_train, y_train):
    st.write("Building Machine Learning Models")
    models = {}
    models['Linear Regression'] = LinearRegression()
    models['Logistic Regression'] = LogisticRegression()
    models['Decision Tree Regressor'] = DecisionTreeRegressor()
    models['Decision Tree Classifier'] = DecisionTreeClassifier()
    models['Random Forest Regressor'] = RandomForestRegressor()
    models['Random Forest Classifier'] = RandomForestClassifier()
    models['Support Vector Regressor'] = SVR()
    models['Support Vector Classifier'] = SVC()
    models['Gaussian Naive Bayes'] = GaussianNB()
    models['Multinomial Naive Bayes'] = MultinomialNB()
    models['Bernoulli Naive Bayes'] = BernoulliNB()
    models['K-Nearest Neighbors'] = KNeighborsClassifier()
    for name, model in models.items():
        model.fit(X_train, y_train)
    return models

# Function to build deep learning models
def build_dl_models(X_train, y_train):
    st.write("Building Deep Learning Models")
    models = {}
    model1 = Sequential()
    model1.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
    model1.add(Dense(32, activation='relu'))
    model1.add(Dense(1))
    model1.compile(optimizer='adam', loss='mean_squared_error')
    model1.fit(X_train, y_train, epochs=10, batch_size=32)
    models['Deep Neural Network'] = model1
    return models

# Function to perform NLP tasks
def perform_nlp_tasks(text):
    st.write("Performing NLP Tasks")
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    st.write("Filtered Tokens:", filtered_tokens)

    # Sentiment analysis
    sentiment_pipeline = pipeline('sentiment-analysis')
    sentiment = sentiment_pipeline(text)
    st.write("Sentiment:", sentiment)

# Function to get LLM response
def get_llm_response(query):
    try:
        response = llm.invoke(query)
        if response:
            return response
        else:
            return "No response generated."
    except Exception as e:
        return str(e)

# Streamlit app
st.title("Aakku AI - Machine Learning and Deep Learning App")
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    data = load_data(uploaded_file)
    explore_data(data)
    data = preprocess_data(data)
    X = data.drop('target', axis=1)
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models = build_ml_models(X_train, y_train)
    dl_models = build_dl_models(X_train, y_train)

    # NLP tasks
    text = st.text_input("Enter text for NLP tasks")
    if text:
        perform_nlp_tasks(text)

    # LLM response
    st.write("Ask Aakku AI anything about machine learning or deep learning")
    user_query = st.text_input("Enter your query")
    if user_query:
        response = get_llm_response(user_query)
        st.write("Aakku AI:", response)
