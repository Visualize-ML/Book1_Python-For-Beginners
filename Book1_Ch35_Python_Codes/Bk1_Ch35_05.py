import streamlit as st

# 在两列中显示不同的内容
col1, col2 = st.columns(2)
col1.write("This is column 1")
col1.latex(r'f(x) = ax^2 + bx + c')

col2.write("This is column 2")