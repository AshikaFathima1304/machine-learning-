import streamlit as st
import matplotlib

# Streamlit app title
st.title("Matplotlib Import Test")

# Check if Matplotlib can be imported
try:
    import matplotlib.pyplot as plt
    st.write("Matplotlib successfully imported")
except Exception as e:
    st.error("Error occurred while importing Matplotlib:")
    st.error(str(e))
