import streamlit as st
# 创建两个选项卡，每个选项卡显示不同的内容
tab_A, tab_B = st.tabs(["Tab A", "Tab B"])

with tab_A:
   st.header("Tab A Title")
   st.write('This is Tab A.')

with tab_B:
   st.header("Tab B Title")
   st.write('This is Tab B.')