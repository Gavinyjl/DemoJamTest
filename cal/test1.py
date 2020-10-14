import numpy as np
import pandas as pd
import sklearn.metrics.pairwise as pw
import sklearn.preprocessing as ppc

msg = 'start py'
print('msg:', msg)


pd.set_option('display.max_columns', 100)
fan = pd.read_csv('cal/alternator_fan.csv')
fan1 = fan.loc[:, ['material ID', 'diameter',
                   'width', 'weight', 'price', 'rank']]
print('fan1:\n', fan1)

print('fan1.shape:', fan1.shape)

fan2 = fan1.iloc[:, 1:6]
print('fan2:', fan2)
print('fan2.mean0:', fan2.mean(axis=0))
print('fan2.std0:', fan2.std(axis=0))


fan2_scaled = ppc.scale(fan2)
print('fan2_scaled:\n', fan2_scaled)
print('fan2_scaled.mean:\n', fan2_scaled.mean(axis=0))
print('fan2_scaled.std:\n', fan2_scaled.std(axis=0))

fan3 = ppc.normalize(fan2, norm='l2')
print('fan3:\n', fan3)

min_max_scaler = ppc.MinMaxScaler(feature_range=(0, 1))
fan2_min_max = min_max_scaler.fit_transform(fan2)
print('fan2_min_max.mean:\n', fan2_min_max.mean(axis=0))
print('fan2_min_max.std:\n', fan2_min_max.std(axis=0))




# df = pd.DataFrame(np.arange(0, 60, 2).reshape(10, 3), columns=list('abc'))
# print('df:\n', df)
# print('df.mean0:\n', df.mean(axis=0))
# print('df.mean1:\n', df.mean(axis=1))
# df1 = df.iloc[5:8, [1, 2]]
# print('df1', df1)
