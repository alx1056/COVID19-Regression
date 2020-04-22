#importing to retrieve data from the internet
import requests
import pandas as pd

url = "https://covidtracking.com/data/state/north-carolina#historical"#website for NC COVID data
html = requests.get(url).content
df_list = pd.read_html(html)
df = df_list[-1]#gets rid of the last column

#creates a new DF without dates
df_1 = pd.DataFrame(df,columns=['New Tests', 'Positive', 'Negative', 'Deaths', 'Total'])
print(df_1)

#need to reverse the order since this is from newest cases to oldest
df_1 = df_1[::-1]

df_1.head()




#Drops NaN values
df_1 = df_1.dropna()

df_1.head()




#importing for cleansing and analysis
import numpy as np
from scipy import stats
from datetime import datetime
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
#keeps graphs stationary inside IDE
%matplotlib inline







#shows first few rows of data in DataFrame
df_1.head()

#Shows Scatter plot of DataFrame
plt.scatter(X,y)
plt.xlabel('Cases')#x-label
plt.ylabel('Days Since *March-15-2020*')#y-label

#Shows line plot of DataFrame
df_1.plot(figsize=(18,5))
plt.xlabel('Days Since *March-15-2020*')#x-label
plt.ylabel('Cases')#y-label




#Linear Regression of COVID19 Data
#Wanting to see positive Correlation with Positive Cases and Deaths
X = pd.DataFrame(df_1['Positive'])
y = pd.DataFrame(df_1['Deaths'])
model = LinearRegression()
model.fit(X, y)
model = LinearRegression().fit(X, y)






#shows R^2, m and b from regression formula
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)

y_pred = model.predict(X)
print('predicted positive cases over time', y_pred, sep='\n')





#Using Kflods to make the data unbiased
kfold = KFold(n_splits=2, shuffle=True, random_state=42)
for i, (train, test) in enumerate(kfold.split(X, y)):
 model.fit(X.iloc[train,:], y.iloc[train,:])
 score = model.score(X.iloc[test,:], y.iloc[test,:])
 scores.append(score)
print(scores)#will print (#K) R^2 values
