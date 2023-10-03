import streamlit as st
import seaborn as sns
import plotly.express as px

# 显示标题
st.title('Welcome to the world of :red[Streamlit]')
# 显示章节标题
st.header('Pandas DataFrame')
# 显示 markdown 文本
st.markdown("Load :blue[Iris Data Set]")
# 从Seaborn导入鸢尾花数据帧
df = sns.load_dataset('iris')
# 显示数据帧
st.write(df)
# 显示章节标题
st.header('Visualize Using Heatmap')
fig = px.imshow(df.iloc[:,:-1])
# 显示热图
st.write(fig)