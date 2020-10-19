import numpy as np
import pandas as pd
import sklearn.metrics.pairwise as pw
import sklearn.preprocessing as ppc
import random
import scipy as sp
from matplotlib import pyplot as plt

msg = 'start py'
print('msg:', msg)

np.set_printoptions(threshold=np.inf, precision=3)


def to_sum_one(arr):
    arr = np.array(arr)
    print('arr的类型：（函数内）', type(arr))
    arr = arr/arr.sum()
    print("\n归一化后的矩阵：", arr)
    return(arr)


pd.set_option('display.max_columns', 100)
fan = pd.read_csv('cal/alternator_fan.csv')
fan1 = fan.loc[:, ['material ID', 'diameter',
                   'width', 'weight', 'price', 'rank']]  # 选取需要的维度
print('fan1:\n', fan1)

print('fan1.shape:', fan1.shape)

# fan2是将fan1的dataframe转化为二维矩阵
# def df_to_matrix(df):
#     arr=df.
fan2 = fan1.iloc[:, 1:6]
print('fan2:\n', fan2)
print('fan2.mean0:', fan2.mean(axis=0))
print('fan2.std0:', fan2.std(axis=0))

# 绘制fan2图像
fig = plt.figure()
index = [i for i in range(10000)]
plt.plot(index, fan2['diameter'], '.')
# plt.show()


# 标准化Z-Score，使均值在0附近，去除方差
# fan2_scaled = ppc.scale(fan2)
# print('fan2_scaled:\n', fan2_scaled)
# print('fan2_scaled.mean:\n', fan2_scaled.mean(axis=0))
# print('fan2_scaled.std:\n', fan2_scaled.std(axis=0))

# 正则化normalization
# fan3 = ppc.normalize(fan2, norm='l2')
# print('fan3:\n', fan3)

# MinMaxScaler将fan2归一化
min_max_scaler = ppc.MinMaxScaler(feature_range=(0, 1))
fan2_min_max = min_max_scaler.fit_transform(fan2)
print('fan2_min_max:\n', fan2_min_max)  # 平均数
print('fan2_min_max.mean:\n', fan2_min_max.mean(axis=0))  # 平均数
print('fan2_min_max.std:\n', fan2_min_max.std(axis=0))  # 标准差

# 用户选择item次数
user_prefer = []
# for i in range(10000):
#     num = random.randint(0, 20)
#     user_prefer.append(num)

row_num = 1
col_num = 10000
ele_num = 500

user_prefer = [row_num-1 for _ in range(col_num)]
# print('row_arr', row_arr)
row_seq = [i for i in range(0, col_num)]
insert_arr = random.sample(row_seq, 100)
print('\ninsert_arr', insert_arr)
for i in range(len(insert_arr)):
    user_prefer[insert_arr[i]] = random.randint(0, 10)
print('\nuser_prefer插值后', user_prefer)

# print('arr的类型：（函数前）', type(user_prefer))
user_prefer = to_sum_one(user_prefer)
# print('arr的类型：（函数后）', type(user_prefer))


# b = sp.sparse.rand(1, 10000, 0.005)
# print('随机矩阵：', b)


# print('user_prefer:', user_prefer)
# up_array = np.array(user_prefer)
# print('up_array:\n', up_array)
# up_array = up_array/up_array.sum()
# print('up_array回归化:\n', up_array)


# user_profile由user_prefer和fan2_min_max矩阵相乘得到
user_profile = user_prefer.dot(fan2_min_max)
print('user_profile:\n', user_profile)


# df例子
# df = pd.DataFrame(np.arange(0, 60, 2).reshape(10, 3), columns=list('abc'))
# print('df:\n', df)
# print('df.mean0:\n', df.mean(axis=0))
# print('df.mean1:\n', df.mean(axis=1))
# df1 = df.iloc[5:8, [1, 2]]
# print('df1', df1)
