import pandas as pd
from sklearn.datasets import load_iris

iris_data = load_iris()
pandas_DF = pd.DataFrame(data=iris_data.data, columns=['sepal_length','sepal_width','petal_length','petal_width'])

print(pandas_DF)
print(type(pandas_DF))
# <class 'pandas.core.frame.DataFrame'>

print(pandas_DF.head(10))
print(pandas_DF.shape)
# (150, 4)

print(pandas_DF.info())
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 150 entries, 0 to 149
# Data columns (total 4 columns):
#  #   Column        Non-Null Count  Dtype
# ---  ------        --------------  -----
#  0   sepal_length  150 non-null    float64
#  1   sepal_width   150 non-null    float64
#  2   petal_length  150 non-null    float64
#  3   petal_width   150 non-null    float64
# dtypes: float64(4)

print(pandas_DF.describe())
#        sepal_length  sepal_width  petal_length  petal_width
# count    150.000000   150.000000    150.000000   150.000000
# mean       5.843333     3.057333      3.758000     1.199333
# std        0.828066     0.435866      1.765298     0.762238
# min        4.300000     2.000000      1.000000     0.100000
# 25%        5.100000     2.800000      1.600000     0.300000
# 50%        5.800000     3.000000      4.350000     1.300000
# 75%        6.400000     3.300000      5.100000     1.800000
# max        7.900000     4.400000      6.900000     2.500000

print(pandas_DF['sepal_width'].value_counts())
# 3.0    26
# 2.8    14
# 3.2    13
# 3.4    12
# 3.1    11
# 2.9    10
# 3.0이 26개있다.

year_feature = pandas_DF['sepal_width']

print(year_feature.head(10))
year_value = pandas_DF['sepal_width'].value_counts()
print(year_value)

print(pandas_DF.columns)
# # Index(['sepal_length', 'sepal_width', 'petal_length', 'petal_width'], dtype='object')