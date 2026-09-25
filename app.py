import streamlit as st 
st.title('my first streamit app')
st.header('welcome!')
st.write('this is a basic app')
name = st.text_input("arin")
if name:
    st.success(f'hey, {name}!')