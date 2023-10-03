import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# 产生数据
x1_array = np.linspace(-3, 3, 301)
x2_array = np.linspace(-3, 3, 301)
xx1, xx2 = np.meshgrid(x1_array, x2_array)

# 二元函数的曲面数据
ff = xx1 * np.exp(-xx1**2 - xx2**2)

# Matplotly图像
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(projection='3d')
ax.plot_wireframe(xx1, xx2, ff, rstride=10, cstride=10)
st.pyplot(fig)

# Plotly图像
fig = go.Figure(data=[go.Surface(z=ff, x=xx1, y=xx2,
                      colorscale = 'RdYlBu_r')])
st.plotly_chart(fig)