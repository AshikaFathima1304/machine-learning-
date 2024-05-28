import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st

st.title('Tennis Play Prediction')

# Upload CSV
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    # Load data
    data = pd.read_csv(uploaded_file)
    st.write("The first 5 values of data are:")
    st.write(data.head())

    # Obtain Train data and Train output
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Convert categorical data into numerical data using LabelEncoder
    le_outlook = LabelEncoder()
    X['Outlook'] = le_outlook.fit_transform(X['Outlook'])

    le_temperature = LabelEncoder()
    X['Temperature'] = le_temperature.fit_transform(X['Temperature'])

    le_humidity = LabelEncoder()
    X['Humidity'] = le_humidity.fit_transform(X['Humidity'])

    le_windy = LabelEncoder()
    X['Windy'] = le_windy.fit_transform(X['Windy'])

    le_play_tennis = LabelEncoder()
    y = le_play_tennis.fit_transform(y)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # Train Gaussian Naive Bayes classifier
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    # Evaluate classifier
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_pred, y_test)

    st.write(f"Accuracy is: {accuracy:.2f}")

else:
    st.write("Please upload a CSV file.")
