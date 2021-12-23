import math
import pandas as pd
from word2number import w2n
from sklearn import linear_model

df = pd.read_csv("hiring.csv")
#print(df)

################### cleaning data ####################
# replacing NaN with average of all cells in column #
##### converting strings to nums for experience #####

median_test_score = df.test_score.median()
#print(median_test_score)

df.experience.fillna("zero", inplace=True)
df.test_score.fillna(median_test_score, inplace=True)

for i in range(len(df.experience)):
    df.at[i, 'experience'] = w2n.word_to_num(df.loc[i].at['experience'])

print(df)

##################### training #######################

reg = linear_model.LinearRegression()
features = df[['experience', 'test_score', 'interview_score']]
label = df.salary

reg.fit(features.values, label)
#print(reg.coef_)
#print(reg.intercept_)

################### predictions #####################

print(f"\nFirst candidate recommended salary: ${math.ceil(reg.predict([[2, 9, 6]]))}/yr.")
print(f"Second candidate recommended salary: ${math.ceil(reg.predict([[12, 10, 10]]))}/yr.")

