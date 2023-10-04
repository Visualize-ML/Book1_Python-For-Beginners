###############
# Authored by Weisheng Jiang
# Book 1  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2023
###############

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import streamlit as st

p = plt.rcParams
p["font.sans-serif"] = ["Roboto"]
p["font.weight"] = "light"
p["ytick.minor.visible"] = True
p["xtick.minor.visible"] = True
p["axes.grid"] = True
p["grid.color"] = "0.5"
p["grid.linewidth"] = 0.5

with st.sidebar:
    st.title('Bivariate Gaussian Distribution')
    # 质心位置
    mu_X1 = st.slider('mu_X1', min_value = -4.0, 
                      max_value = 4.0, 
                        value = 0.0, step = 0.1)
    mu_X2 = st.slider('mu_X2', min_value = -4.0, 
                      max_value = 4.0, 
                      value = 0.0, step = 0.1)
    # 标准差
    sigma_X1 = st.slider('sigma_X1', min_value = 0.5, 
                         max_value = 3.0, 
                        value = 1.0, step = 0.1)
    sigma_X2 = st.slider('sigma_X2', min_value = 0.5, 
                         max_value = 3.0, 
                        value = 1.0, step = 0.1)
    # 相关性系数
    rho = st.slider('rho', min_value = -0.95, 
                    max_value = 0.95, 
                    value = 0.0, step = 0.05)

# 质心    
mu    = [mu_X1, mu_X2]
# 协方差矩阵
Sigma = [[sigma_X1**2, sigma_X1*sigma_X2*rho], 
        [sigma_X1*sigma_X2*rho, sigma_X2**2]]

width = 4
x1 = np.linspace(-width,width,321)
x2 = np.linspace(-width,width,321)

xx1, xx2 = np.meshgrid(x1, x2)

xx12 = np.dstack((xx1, xx2))
bi_norm = multivariate_normal(mu, Sigma)
# 二元高斯PDF
PDF_joint = bi_norm.pdf(xx12)

# 绘制二元高斯PDF等高线
fig, ax = plt.subplots(figsize=(5, 5))

plt.contourf(xx1, xx2, PDF_joint, 20, cmap='RdYlBu_r')
plt.axvline(x = mu_X1, color = 'k', linestyle = '--')
plt.axhline(y = mu_X2, color = 'k', linestyle = '--')

ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')

st.pyplot(fig)
