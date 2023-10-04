###############
# Authored by Weisheng Jiang
# Book 1  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

p = plt.rcParams
p["font.sans-serif"] = ["Roboto"]
p["font.weight"] = "light"
p["ytick.minor.visible"] = True
p["xtick.minor.visible"] = True
p["axes.grid"] = True
p["grid.color"] = "0.5"
p["grid.linewidth"] = 0.5

import pandas_datareader as pdr
# pip install pandas_datareader
import seaborn as sns
import statsmodels.multivariate.pca as pca
import streamlit as st
p = plt.rcParams
p["font.sans-serif"] = ["Roboto"]
p["font.weight"] = "light"
p["ytick.minor.visible"] = True
p["xtick.minor.visible"] = True
p["axes.grid"] = True
p["grid.color"] = "0.5"
p["grid.linewidth"] = 0.5

df = pdr.data.DataReader(['DGS6MO','DGS1',
                          'DGS2','DGS5',
                          'DGS7','DGS10',
                          'DGS20','DGS30'], 
                          data_source='fred', 
                          start='01-01-2022', 
                          end='12-31-2022')

df = df.dropna()

df = df.rename(columns={'DGS6MO': '0.5 yr', 
                        'DGS1': '1 yr',
                        'DGS2': '2 yr',
                        'DGS5': '5 yr',
                        'DGS7': '7 yr',
                        'DGS10': '10 yr',
                        'DGS20': '20 yr',
                        'DGS30': '30 yr'})
X_df = df.pct_change(); 
# convert level to daily difference
X_df = X_df.dropna()

with st.sidebar:
    st.title('Principal Component Analysis')
    num_of_PCs = st.slider('Number of PCs',
             min_value = 1, 
             max_value = 8, 
             value = 2, step = 1)
    
pca_model = pca.PCA(X_df, standardize=True)
variance_V = pca_model.eigenvals 
# 计算主成分的方差解释比例
explained_var_ratio = pca_model.eigenvals / pca_model.eigenvals.sum()

X_df_ = pca_model.project(num_of_PCs)

fig, axes = plt.subplots(2,4,figsize=(8,4))
axes = axes.flatten()

for col_idx, ax_idx in zip(list(X_df_.columns),axes):
    sns.lineplot(X_df_[col_idx],ax = ax_idx)
    sns.lineplot(X_df[col_idx],ax = ax_idx) 
    sns.lineplot(X_df[col_idx] - X_df_[col_idx], c = 'k', ax = ax_idx) 
    ax_idx.set_xticks([])
    ax_idx.set_yticks([])
    ax_idx.axhline(y = 0, c = 'k')
    # ax_idx.plot([-0.3, 0.3],[-0.3, 0.3],c = 'r')
    # ax_idx.set_aspect('equal', adjustable='box')
    # ax_idx.set_xlim(-0.3, 0.3)
    # ax_idx.set_ylim(-0.3, 0.3)
plt.tight_layout()

st.pyplot(fig)