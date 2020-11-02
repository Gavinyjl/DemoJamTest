from test1 import CBRecommendation
import pandas as pd
import sklearn.preprocessing as ppc
import random
from matplotlib import pyplot as plt


if __name__ == "__main__":
    cbr = CBRecommendation()
    pd.set_option('display.max_columns', 100)
    fan = pd.read_csv('cal/alternator_fan.csv')
    fan1 = fan.loc[:, ['material ID', 'diameter',
                       'width', 'weight', 'price', 'rank']]  # 选取需要的维度
    print('fan1:\n', fan1)

    print('fan1.shape:', fan1.shape)

    # fan2是将fan1的dataframe转化为二维矩阵
    fan2 = fan1.iloc[:, 1:6]
    cbr.print_fan2(fan2)

    cbr.fig_show(fan2['price'])

    # 标准化Z-Score，使均值在0附近，去除方差
    # fan2_scaled = ppc.scale(fan2)
    # print('fan2_scaled:\n', fan2_scaled)
    # print('fan2_scaled.mean:\n', fan2_scaled.mean(axis=0))
    # print('fan2_scaled.std:\n', fan2_scaled.std(axis=0))

    # 正则化normalization
    # fan3 = ppc.normalize(fan2, norm='l2')
    # print('fan3:\n', fan3)

    """ MinMaxScaler将fan2归一化，使特征分布在0-1之间 """
    min_max_scaler = ppc.MinMaxScaler(feature_range=(0, 1))
    fan2_min_max = min_max_scaler.fit_transform(fan2)

    cbr.print_fan2_min_max(fan2_min_max)

    # b = sp.sparse.rand(1, 10000, 0.005)
    # print('随机矩阵：', b)

    # print('user_prefer:', user_prefer)
    # up_array = np.array(user_prefer)
    # print('up_array:\n', up_array)
    # up_array = up_array/up_array.sum()
    # print('up_array回归化:\n', up_array)

    """ 创建user_profile """
    user_profile = []
    user_profile = cbr.create_user_profile(fan2_min_max, 10, 20)
    # profile_test(fan2_min_max)

    # user_prefer = process_up(10, 10)
    # # user_profile由user_prefer和fan2_min_max矩阵相乘得到
    # user_profile = user_prefer.dot(fan2_min_max)
    # print('user_profile:\n', user_profile)

    # df
    # df = pd.DataFrame(np.arange(0, 60, 2).reshape(10, 3), columns=list('abc'))
    # print('df:\n', df)
    # print('df.mean0:\n', df.mean(axis=0))
    # print('df.mean1:\n', df.mean(axis=1))
    # df1 = df.iloc[5:8, [1, 2]]
    # print('df1', df1)
