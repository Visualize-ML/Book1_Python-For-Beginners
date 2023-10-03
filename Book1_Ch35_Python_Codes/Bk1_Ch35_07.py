import streamlit as st
import seaborn as sns
import plotly.express as px

# 显示标题
st.title('Iris Dataset')

# 从Seaborn导入鸢尾花数据帧
df = sns.load_dataset('iris')
# 第一个可展开区域
with st.expander("Open and view DataFrame"):
    # 显示数据帧
    st.write(df)
# 第二个可展开区域
with st.expander("Open and view Heatmap"):
    fig = px.imshow(df.iloc[:,:-1])
    # 显示热图
    st.write(fig)