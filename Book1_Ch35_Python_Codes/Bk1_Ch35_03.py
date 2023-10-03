import streamlit as st
button_return = st.button("Click me")
st.write(button_return)
st.checkbox("Check me")
st.radio("Choose one:", 
         ["A", "B", "C"])
st.selectbox("Choose one:", 
             ["A", "B", "C"])
st.multiselect("Choose many:", 
               ["A", "B", "C", "D"])
st.slider("Select a value:", 
          0.0, 10.0, 5.0)
st.select_slider("Select a value:", 
                 options=[1, 2, 3, 4, 5])
st.text_input("Enter your name")
st.number_input("Enter a number")
st.text_area("Enter your message")
st.date_input("Select a date")
st.time_input("Select a time")
st.file_uploader("Upload a file")
st.color_picker("Pick a color")