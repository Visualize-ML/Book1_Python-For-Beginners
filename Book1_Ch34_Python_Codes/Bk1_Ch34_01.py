
# =============================================================================
# 快捷键 ctrl + 4 产生这部分注释模板
# =============================================================================


#%% 导入库
# =============================================================================
# 导入库
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%% 等差数列
# =============================================================================
# 等差数列
# =============================================================================

a0 = 1 # 首项
n = 10 # 项数
d = 2 # 公差

a_array = np.arange(a0, a0 + n*d, d)
print('打印等差数列'); print(a_array)

fig = plt.figure(figsize = (8,8))
plt.scatter(np.arange(n), a_array)
plt.title('Arithmetic Progression')
plt.xlabel('Index, $n$')
plt.ylabel('Value, $a_n$')
plt.show()
#%% 二元函数
# =============================================================================
# 二元函数
# =============================================================================

x1_array = np.linspace(-3, 3, 301)
x2_array = np.linspace(-3, 3, 301)
xx1, xx2 = np.meshgrid(x1_array, x2_array)
# np.meshgrid() 函数将 x1_array 和 x2_array 两个一维数组转化为一个二维的网格点矩阵

ff = xx1 * np.exp(-xx1**2 - xx2**2)
# 二元函数的曲面数据

# 可视化
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(projection='3d')
# 绘制二元函数网格曲面
ax.plot_wireframe(xx1, xx2, ff, rstride=10, cstride=10)
plt.show()
#%%
# =============================================================================
# 鸢尾花数据
# =============================================================================

# 加载鸢尾花数据集
iris_df = sns.load_dataset('iris')
# 显示数据集前5行
print('打印鸢尾花数据前5行'); print(iris_df.head())

#%%%
fig, ax = plt.subplots(figsize = (8,8))
ax = sns.scatterplot(data=iris_df, x="sepal_length", 
                     y="sepal_width", hue = "species")
# hue 用不同色调表达鸢尾花的类别

ax.set_xlabel('Sepal length (cm)')
ax.set_ylabel('Sepal width (cm)')
ax.set_xticks(np.arange(4, 8 + 1, step=1))
ax.set_yticks(np.arange(1, 5 + 1, step=1))
ax.axis('scaled')
ax.grid(linestyle='--', linewidth=0.25, color=[0.7,0.7,0.7])
ax.set_xbound(lower = 4, upper = 8)
ax.set_ybound(lower = 1, upper = 5)

#%%