

import streamlit as st
import plotly.figure_factory as ff
from scipy.stats import dirichlet
import numpy as np
import matplotlib.tri as tri

with st.sidebar:
    st.title('Dirichlet distribution')
    alpha_1 = st.slider('alpha_1', 1.0, 5.0, step = 0.1, value = 2.0)
    alpha_2 = st.slider('alpha_2', 1.0, 5.0, step = 0.1, value = 2.0)
    alpha_3 = st.slider('alpha_3', 1.0, 5.0, step = 0.1, value = 2.0)
    
    alpha_array = [alpha_1,alpha_2,alpha_3]

# 定义等边三角形
corners = np.array([[0, 0], [1, 0], [0.5,0.75**0.5]]).T
triangle = tri.Triangulation(corners[0,:], corners[1,:])
refiner = tri.UniformTriRefiner(triangle)
trimesh_5 = refiner.refine_triangulation(subdiv=5)

# 自定义函数
def xy2bc(trimesh_8):
    
    # 每个列向量代表一个三角网格坐标点
    r_array = np.row_stack((trimesh_5.x,trimesh_5.y))

    r1 = corners[:,[0]]
    r2 = corners[:,[1]]
    r3 = corners[:,[2]]

    T = np.column_stack((r1 - r3,r2 - r3))

    theta_1_2 = np.linalg.inv(T) @ (r_array - r3)
    theta_3 = 1 - theta_1_2[0,:] - theta_1_2[1,:]

    theta_1_2_3 = np.row_stack((theta_1_2,theta_3))
    theta_1_2_3 = np.clip(theta_1_2_3, 1e-6, 1.0 - 1e-6)
    theta_1_2_3 = theta_1_2_3/theta_1_2_3.sum(axis = 0) 
    # 归一化
    
    return theta_1_2_3

tri_coordinates = xy2bc(trimesh_5)
PDF = dirichlet.pdf(tri_coordinates, alpha_array)

fig = ff.create_ternary_contour(tri_coordinates, PDF,
                                pole_labels=['theta_3',
                                             'theta_1',
                                             'theta_2'],
                                interp_mode='cartesian',
                                ncontours=12,
                                colorscale='Viridis',
                                showscale=True,)
st.plotly_chart(fig, use_container_width=True)