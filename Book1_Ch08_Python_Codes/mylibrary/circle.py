import math

def area(radius):
    '''
    计算圆面积
    输入：
    radius：半径
    输出：
    area：面积
    '''
    
    area = math.pi * radius**2
    return area

def circumference(radius):
    '''
    计算圆周长
    输入：
    radius：半径
    输出：
    circ：周长
    '''
    
    circ = 2 * math.pi * radius
    return circ

