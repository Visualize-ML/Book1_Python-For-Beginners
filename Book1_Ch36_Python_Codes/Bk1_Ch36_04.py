###############
# Authored by Weisheng Jiang
# Book 1  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import streamlit as st

p = plt.rcParams
p["font.sans-serif"] = ["Roboto"]
p["font.weight"] = "light"
p["ytick.minor.visible"] = True
p["xtick.minor.visible"] = True
p["axes.grid"] = True
p["grid.color"] = "0.5"
p["grid.linewidth"] = 0.5

# 生成随机数据
np.random.seed(0)
num = 30
X = np.random.uniform(0,4,num)
y = np.sin(0.4*np.pi * X) + 0.4 * np.random.randn(num)
data = np.column_stack([X,y])

x_array = np.linspace(0,4,101).reshape(-1,1)
degree_array = [1,2,3,4,7,8]

with st.sidebar:
    st.title('Polynomial Regression')
    degree = st.slider('Degree',
             min_value = 1, 
             max_value = 9, 
             value = 2, step = 1)

    
    
fig, ax = plt.subplots(figsize=(5,5))


poly = PolynomialFeatures(degree = degree)
X_poly = poly.fit_transform(X.reshape(-1, 1))

# 训练线性回归模型
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_poly_pred = poly_reg.predict(X_poly)
data_ = np.column_stack([X,y_poly_pred])

y_array_pred = poly_reg.predict(
                   poly.fit_transform(x_array))

# 绘制散点图
ax.scatter(X, y, s=20)
ax.scatter(X, y_poly_pred, marker = 'x', color='k')

ax.plot(([i for (i,j) in data_], [i for (i,j) in data]),
        ([j for (i,j) in data_], [j for (i,j) in data]),
         c=[0.6,0.6,0.6], alpha = 0.5)

ax.plot(x_array, y_array_pred, color='r')

# 提取参数
coef = poly_reg.coef_
intercept = poly_reg.intercept_
# 回归解析式
equation = '$y = {:.1f}'.format(intercept)
for j in range(1, len(coef)):
    equation += ' + {:.1f}x^{}'.format(coef[j], j)
equation += '$'
equation = equation.replace("+ -", "-")
# ax.text(0.05, -1.8, equation)
st.write(equation)
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(0,4)
ax.grid(False)
ax.set_ylim(-2,2)

st.pyplot(fig)