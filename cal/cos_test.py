import math
import numpy as np


def dotProduct(v1, v2):
    """
    点乘
    """
    v1 = np.mat(v1)
    v2 = np.mat(v2)
    m = v1.dot(v2.T)
    print('点乘结果：', m)
    return m


def cosSim(v1, v2):
    """
    余弦相似度
    """
    sim = dotProduct(v1, v2)/math.sqrt(dotProduct(v1, v1)*dotProduct(v2, v2))
    print('余弦相似度：', sim)
    return sim


def toMatrix(x):
    """
    数组矩阵化
    """
    x = np.mat(x)
    return x


def eucDist(x, y):
    """
    计算欧氏距离
    """
    dist = np.linalg.norm(toMatrix(x)-toMatrix(y))
    print('欧氏距离：', dist)
    return dist


x = [1, 1, 1, 1]
y = [2, 2, 2, 4]
cosSim(x, y)

eucDist(x, y)
