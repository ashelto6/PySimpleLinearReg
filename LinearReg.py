import math
import numpy as mp
import pandas as pd
from sklearn import linear_model

df = pd.read_csv("homeprices.csv")

################### cleaning data ####################
# replacing NaN with average of all cells in column #
median_bedrooms = math.ceil(df.bedrooms.median())
#print(median_bedrooms) # 3

df.bedrooms.fillna(median_bedrooms, inplace=True)
#print(df)

################### model training ###################

reg = linear_model.LinearRegression()
features = df[['area', 'bedrooms', 'age']]
label = df.price
reg.fit(features.values, label)

#print(reg.coef_)
#print(reg.intercept_)

print(math.ceil(reg.predict([[3000, 3, 40]])))