import streamlit as st
import numpy as np
from sympy import symbols,lambdify
import matplotlib.pyplot as plt

# 侧边框
with st.sidebar:
    st.header('Choose coefficients')
    st.latex(r'f(x) = ax^2 + bx + c')
    a = st.slider("a",min_value = -5.0, 
                  max_value = 5.0,
                  step = 0.01, value = 1.0)
    b = st.slider("b",min_value = -5.0, 
                  max_value = 5.0,
                  step = 0.01, value = -2.0)
    c = st.slider("c",min_value = -5.0, 
                  max_value = 5.0,
                  step = 0.01, value = -3.0)
# 抛物线
x = symbols('x')
f_x = a*x**2 + b*x + c
x_array = np.linspace(-5,5,101)
f_x_fcn = lambdify(x, f_x)
y_array = f_x_fcn(x_array)

# 主页面
st.title('Qudratic function')
st.latex(r'f(x) = ')
st.latex(f_x)

# 可视化
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x_array, y_array)

ax.set_xlim([-5, 5])
ax.set_ylim([-5, 5])
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
st.write(fig)