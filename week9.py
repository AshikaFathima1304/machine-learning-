import streamlit as st
import pandas as pd

# Load your data
@st.cache
def load_data():
    # Load your data here, for example:
    data = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [6, 7, 8, 9, 10]
    })
    return data

def main():
    st.title("Your Streamlit App")

    # Load data
    data = load_data()

    # Display data
    st.subheader("Data Overview")
    st.write(data.head())

if __name__ == "__main__":
    main()
