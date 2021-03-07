import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.datasets import load_boston

print("*"*50)
print("Read Csv file")
print("*"*50)
df = pd.read_csv("Social_Network_Ads.csv")

print("*"*50)
print("Loading boston data from scikit learn library")
print("*"*50)
boston_data = load_boston()
df_boston = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)
print(df_boston.head())

print("*"*50)
print("Create a dataset and remove the last column")
print("*"*50)
X = df
y = boston_data.target

print("*"*50)
print("Use Stats model for statistical analysis of the data by calling OLS method")
print("*"*50)
X_constant = sm.add_constant(X)
X_constant = pd.DataFrame(X_constant)
model = sm.OLS(y,X_constant)
lr = model.fit()
lr.summary()

