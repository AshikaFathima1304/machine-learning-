# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data from CSV (adjust the path accordingly)
file_path = 'C:\\Users\\ashik\\OneDrive\\Desktop\\ID3\\tennis.csv'  # Adjust the path if necessary

try:
    data = pd.read_csv(file_path)
    print("The first 5 values of data are:\n", data.head())
except FileNotFoundError as e:
    print(f"Error: {e}. Please check the file path and ensure the file exists.")

# Proceed with the rest of the script only if data is loaded successfully
if 'data' in locals():
    # Obtain Train data and Train output
    X = data.iloc[:, :-1]
    print("\nThe First 5 values of train data are:\n", X.head())

    y = data.iloc[:, -1]
    print("\nThe first 5 values of Train output are:\n", y.head())

    # Convert categorical data into numerical data using LabelEncoder
    le_outlook = LabelEncoder()
    X['Outlook'] = le_outlook.fit_transform(X['Outlook'])

    le_temperature = LabelEncoder()
    X['Temperature'] = le_temperature.fit_transform(X['Temperature'])

    le_humidity = LabelEncoder()
    X['Humidity'] = le_humidity.fit_transform(X['Humidity'])

    le_windy = LabelEncoder()
    X['Windy'] = le_windy.fit_transform(X['Windy'])

    print("\nNow the Train data is:\n", X.head())

    le_play_tennis = LabelEncoder()
    y = le_play_tennis.fit_transform(y)
    print("\nNow the Train output is:\n", y)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # Train Gaussian Naive Bayes classifier
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    # Evaluate classifier
    y_pred = classifier.predict(X_test)
    print("Accuracy is:", accuracy_score(y_pred, y_test))
