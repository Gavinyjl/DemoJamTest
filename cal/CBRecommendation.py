import numpy as np
import pandas as pd
import sklearn.metrics.pairwise as pw
import sklearn.preprocessing as ppc
import random
from matplotlib import pyplot as plt


msg = 'start py'
print('msg:', msg)

# 设置numpy在控制台全部输出，inf是无穷大，precision保留3位小数
np.set_printoptions(threshold=np.inf, precision=3)


class CBRecommendation():

    def print_fan2(self, fan2):
        """
        输出fan2参数
        """
        print('fan2:\n', fan2)
        print('fan2.mean0:', fan2.mean(axis=0))
        print('fan2.std0:', fan2.std(axis=0))
        print('输出diameter维度:\n', fan2['diameter'])

    def print_fan2_min_max(self, fan2_min_max):
        """
        输出fan2归一化后的参数
        """
        print('fan2_min_max:\n', fan2_min_max)  # 平均数
        print('fan2_min_max.mean:\n', fan2_min_max.mean(axis=0))  # 平均数
        print('fan2_min_max.std:\n', fan2_min_max.std(axis=0))  # 标准差

    def fig_show(self, fan2):
        """ 绘制fan2图像 """
        # fig = plt.figure()
        index = [i for i in range(10000)]
        plt.plot(index, fan2, '.')
        plt.show()

    def to_sum_one(self, arr):
        """ 将一维矩阵归一化，使所有水元素的和为1 """
        arr = np.array(arr)
        # print('arr的类型：（函数内）', type(arr))
        arr = arr/arr.sum()
        # print("\n归一化后的矩阵：", arr)
        return(arr)

    def init_random_prefer(self, itemNum, maxChooseTime):
        """ 创建一个随机的user_prefer，控制item非零个数和最大选择次数 """
        # user_prefer的行数与列数
        row_num = 1
        col_num = 10000
        # 创建一维零矩阵user_prefer
        user_prefer = [row_num-1 for _ in range(col_num)]
        row_seq = [i for i in range(0, col_num)]  # row_seq是0-9999递增的数组
        # insert_arr是随机选取itemNum个位置作为user_prefer插入位
        insert_arr = random.sample(row_seq, itemNum)
        # print('\ninsert_arr', insert_arr)

        # 在insert_arr选取的插入位中插入0-maxChooseTime之间的随机次数
        for i in range(len(insert_arr)):
            user_prefer[insert_arr[i]] = random.randint(0, maxChooseTime)
        # print('\nuser_prefer插值后', user_prefer)
        return user_prefer

    def process_up(self, itemNum, maxChooseTime):
        """ 生成并归一化user_prefer """
        user_prefer = self.init_random_prefer(itemNum, maxChooseTime)
        # print('user_prefer插值结果\n', user_prefer)

        # print('arr的类型：（函数前）', type(user_prefer))
        user_prefer = self.to_sum_one(user_prefer)
        # print('arr的类型：（函数后）', type(user_prefer))
        return user_prefer

    def create_user_profile(self, fan2_min_max, itemNum, maxChooseTime):
        """
        计算user_profile
        """
        user_prefer = self.process_up(itemNum, maxChooseTime)
        # user_profile由user_prefer和fan2_min_max矩阵相乘得到
        user_profile = user_prefer.dot(fan2_min_max)
        print('[', itemNum, ']', '[', maxChooseTime, ']: ', user_profile)
        return user_profile

    def profile_test(self, fan2_min_max):
        """
        item和次数在[10,100]间遍历计算user_profile
        """
        user_prefer = []
        for i in range(10, 105, 5):
            for j in range(10, 105, 5):
                user_prefer = self.process_up(i, j)
                # user_profile由user_prefer和fan2_min_max矩阵相乘得到
                user_profile = user_prefer.dot(fan2_min_max)
                print('[', i, ']', '[', j, ']: ', user_profile)


