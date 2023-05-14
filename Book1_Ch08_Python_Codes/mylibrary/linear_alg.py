# 自定义函数计算矩阵乘法

def matrix_multiplication(A,B):

    '''
    自定义函数计算两个矩阵乘法
    输入：
    A：矩阵，类型为数据列表
    B：矩阵，类型为数据列表
    输出：
    C：矩阵，类型为数据列表
    参考：https://mathworld.wolfram.com/MatrixMultiplication.html
    '''    
    # 检查两个矩阵是否可以相乘
    if len(A[0]) != len(B):
        raise ValueError("Error: Matrix sizes don't match")

    else:
        # 定义全 0 矩阵 C 用来存放结果
        C = [[0] * len(B[0]) for i in range(len(A))]

        # 遍历 A 的行
        for i in range(len(A)): # len(A) 给出 A 的行数

            # 遍历 B 的列
            for j in range(len(B[0])):  
            # len(B[0]) 给出 B 的列数

                # 这一层相当于消去 p 所在的维度，即压缩
                for k in range(len(B)):
                    C[i][j] += A[i][k] * B[k][j]
                    # 完成对应元素相乘，再求和

        return C

# 计算向量内积
def inner_prod(a,b):
    
    '''
    自定义函数计算两个向量内积
    输入：
    a：向量，类型为数据列表
    b：向量，类型为数据列表
    输出：
    c：标量
    参考：https://mathworld.wolfram.com/InnerProduct.html
    '''

    # 检查两个向量元素数量是否相同
    if len(a) != len(b):
        raise ValueError("Error: check vector lengths")
        
    # 初始化内积为0
    dot_product = 0
    # 使用for循环计算内积
    for i in range(len(a)):
        dot_product += a[i] * b[i]
        
    return dot_product