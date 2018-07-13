#coding=utf-8
import pandas as pd

#创建series对象
pd.Series(['San Francisco', 'San Jose', 'Sacramento'])

city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])

pd.DataFrame({ 'City name': city_names, 'Population': population })

california_housing_dataframe = pd.read_csv("california_housing_train.csv", sep=",")
california_housing_dataframe.describe()
california_housing_dataframe.hist('housing_median_age')