import numpy as np
import pandas as pd


msg = 'start py'
print('msg:', msg)


pd.set_option('display.max_columns', 100)
fan = pd.read_csv('cal/alternator_fan.csv')
fan1 = fan.loc[:, ['material ID', 'diameter',
                   'width', 'weight', 'price', 'rank']]
print('fan1:\n', fan1)

print('shape:', fan1.shape)
# head = fan.head()
# data1 = fan.iloc[0:3, 2]
# print('data1:', data1)

# print(head)


# df = pd.DataFrame(np.arange(0, 60, 2).reshape(10, 3), columns=list('abc'))
# print('df:\n', df)
# df1 = df.iloc[5:8, [1, 2]]
# print('df1', df1)
