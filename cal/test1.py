import pandas as pd
import numpy as np



msg = 'dddsss'
print('msg: ', msg)


pd.set_option('display.max_columns', 100)
df = pd.read_csv('cal/alternator_fan.csv')
head=df.head()

print(head)

