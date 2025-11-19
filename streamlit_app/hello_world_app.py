import streamlit as st

st.title("Hello world App")

st.write("Hello! This is a super simple Streamlit app.")

name = st.text_input("What's your name?")
if name:
    st.success(f"Nice to meet you, {name}!")