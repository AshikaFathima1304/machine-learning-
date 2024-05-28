import streamlit as st
import pandas as pd

def load_data():
    try:
        return pd.read_csv("coronadata.csv")
    except FileNotFoundError:
        st.error("Error: Dataset file not found.")
        return None

def preprocess_data(data):
    # Check if data is not None
    if data is not None:
        # Drop any rows with missing values
        data.dropna(inplace=True)
    return data

def main():
    st.title("CORONA Infection Diagnosis")
    
    # Load data
    data = load_data()
    if data is not None:
        st.subheader("Dataset")
        st.write(data)
        
        # Checkbox to toggle browsing mode
        browse_data = st.checkbox("Browse Dataset")
        
        if browse_data:
            # Show a slider for selecting the number of rows to display
            num_rows = st.slider("Number of Rows", min_value=1, max_value=len(data), value=10)
            st.write(data.head(num_rows))

        # Preprocess data
        data = preprocess_data(data)
        st.subheader("Preprocessed Dataset")
        st.write(data)
    
    # User input for symptoms
    st.subheader("Enter Symptoms")
    fever = st.checkbox("Fever")
    cough = st.checkbox("Cough")
    breathlessness = st.checkbox("Shortness of Breath")
    fatigue = st.checkbox("Fatigue")
    body_aches = st.checkbox("Body Aches")
    loss_of_taste_smell = st.checkbox("Loss of Taste/Smell")
    
    # Analyze symptoms
    if fever or cough or breathlessness or fatigue or body_aches or loss_of_taste_smell:
        st.subheader("Diagnosis")
        st.write("Based on the symptoms entered, please consult a healthcare professional for further evaluation.")
    else:
        st.subheader("Diagnosis")
        st.write("No symptoms selected. Please select at least one symptom.")

if __name__ == "__main__":
    main()
