import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

print("*"*50)
print("Read Csv file")
print("*"*50)
df = pd.read_csv("Social_Network_Ads.csv")

print("*"*50)
print("Check count for missing values in each column")
print("*"*50)
print(df.isnull().sum())

print("*"*50)
print("Create a dataset and remove the last column")
print("*"*50)
X = df.drop(["Purchased","Gender","User ID"],1)
y = df["Purchased"]
print(X)
print(y)

print("*"*50)
print("Seaborn pairplot")
print("*"*50)
# df.drop(["Gender","User ID"],1, inplace=True)
# sns.pairplot(df)

print("*"*50)
print("Seaborn heatmap")
print("*"*50)
df.drop(["Gender","User ID"],1, inplace=True)
sns.heatmap(df.corr(),annot=True)
plt.show()