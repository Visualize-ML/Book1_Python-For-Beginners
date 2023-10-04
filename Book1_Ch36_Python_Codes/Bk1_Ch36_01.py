###############
# Authored by Weisheng Jiang
# Book 1  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2023
###############

import streamlit as st


from scipy.stats import beta
import matplotlib.pyplot as plt
import numpy as np

p = plt.rcParams
p["font.sans-serif"] = ["Roboto"]
p["font.weight"] = "light"
p["ytick.minor.visible"] = True
p["xtick.minor.visible"] = True
p["axes.grid"] = True
p["grid.color"] = "0.5"
p["grid.linewidth"] = 0.5

def uni_normal_pdf(x,mu,sigma):
    
    coeff = 1/np.sqrt(2*np.pi)/sigma
    z = (x - mu)/sigma
    f_x = coeff*np.exp(-1/2*z**2)
    
    return f_x

x_array = np.linspace(-5,5,200)

with st.sidebar:
    st.title('Univariate Gaussian distribution PDF')
    st.latex(r'''{\displaystyle f(x)={\frac {1}{\sigma {\sqrt {2\pi }}}}
             e^{-{\frac {1}{2}}\left({
             \frac {x-\mu }{\sigma }}\right)^{2}}}''')      
    mu_input = st.slider('mu', min_value=-5.0, max_value=5.0, value=0.0, step=0.2)
    sigma_input = st.slider('sigma', min_value=0.0, max_value=4.0, value=1.0, step=0.1)
    
    

pdf_array = uni_normal_pdf(x_array, mu_input, sigma_input)


fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(x_array, pdf_array,
        'b', lw=1)

ax.axvline (x = mu_input, c = 'r', ls = '--')
ax.axvline (x = mu_input + sigma_input, c = 'r', ls = '--')
ax.axvline (x = mu_input - sigma_input, c = 'r', ls = '--')

# standard normal
ax.plot(x_array, uni_normal_pdf(x_array, 0, 1),
        c = [0.8, 0.8, 0.8], lw=1)

ax.axvline (x = 0, c = [0.8, 0.8, 0.8], ls = '--')
ax.axvline (x = 0 + 1, c = [0.8, 0.8, 0.8], ls = '--')
ax.axvline (x = 0 - 1, c = [0.8, 0.8, 0.8], ls = '--')

ax.set_xlim(-5,5)
ax.set_ylim(0,1)
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$f_X(x)$')
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.tick_params(axis="x", direction='in')
ax.tick_params(axis="y", direction='in')

st.pyplot(fig)