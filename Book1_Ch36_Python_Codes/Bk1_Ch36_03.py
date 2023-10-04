###############
# Authored by Weisheng Jiang
# Book 1  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2023
###############

import plotly.graph_objects as go
import numpy as np
import streamlit as st
from scipy.stats import multivariate_normal

st.latex(r'''{\displaystyle f_{\mathbf {X} }(x_{1},\ldots ,x_{k})=
         {\frac {\exp \left(-{\frac {1}{2}}
         ({\mathbf {x} }-{\boldsymbol {\mu }})
         ^{\mathrm {T} }{\boldsymbol {\Sigma }}^{-1}
         ({\mathbf {x} }-{\boldsymbol {\mu }})\right)}
         {\sqrt {(2\pi )^{k}|{\boldsymbol {\Sigma }}|}}}}''')

def bmatrix(a):
    """Returns a LaTeX bmatrix

    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    if len(a.shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')
    lines = str(a).replace('[', '').replace(']', '').splitlines()
    rv = [r'\begin{bmatrix}']
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
    rv +=  [r'\end{bmatrix}']
    return '\n'.join(rv)

# x1_array = np.linspace(-3,3,301)
# x2_array = np.linspace(-3,3,301)
# x3_array = np.linspace(-3,3,301)

xxx1,xxx2,xxx3 = np.mgrid[-3:3:0.2,-3:3:0.2,-3:3:0.2] 

with st.sidebar:
    st.title('Trivariate Gaussian Distribution')
    sigma_1 = st.slider('sigma_1', min_value=0.5, max_value=3.0, value=1.0, step=0.1)
    sigma_2 = st.slider('sigma_2', min_value=0.5, max_value=3.0, value=1.0, step=0.1)
    sigma_3 = st.slider('sigma_3', min_value=0.5, max_value=3.0, value=1.0, step=0.1)
    
    rho_1_2 = st.slider('rho_1_2', min_value=-0.95, max_value=0.95, value=0.0, step=0.05)
    rho_1_3 = st.slider('rho_1_3', min_value=-0.95, max_value=0.95, value=0.0, step=0.05)
    rho_2_3 = st.slider('rho_2_3', min_value=-0.95, max_value=0.95, value=0.0, step=0.05)
    
SIGMA = np.array([[sigma_1**2, rho_1_2*sigma_1*sigma_2, rho_1_3*sigma_1*sigma_3],
                  [rho_1_2*sigma_1*sigma_2, sigma_2**2, rho_2_3*sigma_2*sigma_3],
                  [rho_1_3*sigma_1*sigma_3, rho_2_3*sigma_2*sigma_3, sigma_3**2]])

st.latex(r'\Sigma = ' + bmatrix(SIGMA))

MU = np.array([0, 0, 0])

# st.write(xxx1.shape)
pos = np.dstack((xxx1.ravel(),xxx2.ravel(),xxx3.ravel()))

# st.write(pos.shape)
rv  = multivariate_normal(MU, SIGMA)
PDF = rv.pdf(pos)


fig = go.Figure(data=go.Volume(
    x=xxx1.flatten(),
    y=xxx2.flatten(),
    z=xxx3.flatten(),
    value=PDF.flatten(),
    isomin=0,
    isomax=PDF.max(),
    colorscale = 'RdYlBu_r',
    opacity=0.1, 
    surface_count=11, 
    ))

fig.update_layout(scene = dict(
                    xaxis_title=r'x_1',
                    yaxis_title=r'x_2',
                    zaxis_title=r'x_3'),
                    width=1000,
                    margin=dict(r=20, b=10, l=10, t=10))

st.plotly_chart(fig, theme=None, use_container_width=True)