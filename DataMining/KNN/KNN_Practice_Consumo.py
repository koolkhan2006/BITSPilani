import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor

print("*"*50)
print("Read Csv file")
print("*"*50)
df = pd.read_csv("Consumo_cerveja.csv")
print(df)

print("*"*50)
print("Work on first 10 rows")
print("*"*50)
print(df.head())

print("*"*50)
print("Before working on numerical values ensure there is no null or NAN value")
print("*"*50)
print(df.isnull().sum())
df = df.dropna()
print(df.shape)
print(df.dtypes)
print(df.isnull().sum())

print("*"*50)
print("Change the column names")
print("*"*50)
df.columns = ["Date","Temp(med)", "Temp(min)", "Temp(Max)", "Rainfall(mm)", "weekend", "consumption"]
print(df.head())

print("*"*50)
print("Change the Date column")
print("*"*50)
df.drop(["Date"],1,inplace = True)
print(df.head())

print("*"*50)
print("Loop over list of columns and replace , with .")
print("*"*50)
for x in list(df)[:4]:
    df[x] = df[x].str.replace(",", ".").astype(float)
print(df)


print("*"*50)
print("Change weekend column to category")
print("*"*50)
df["weekend"] = df["weekend"].astype("category")

print("*"*50)
print("We need to calculate the consumption so take it as Y axis and rest of the columns will be used for X axis")
print("*"*50)
X = df.drop(["consumption"],1)
y = df["consumption"]
print(X)
print(y)

print("*"*50)
print("Do the train test split to check the model accuracy for KNN algorithm using mean squared error method")
print("*"*50)
X_train , X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state = 21)
knn = KNeighborsRegressor()
knn.fit(X_train,y_train)
y_pred =  knn.predict(X_test)
print(mean_squared_error(y_test,y_pred)) #best value is 0 because difference between actual and predicted value should be 0 and hence its mean squared will also be 0

print("*"*50)
print("knn.score between test values and predicted values from the model")
print("*"*50)
print(knn.score(X_test,y_test)) # knn.score internally uses the r2 score

print("*"*50)
print("R^2 score Co efficient of determination between test values and predicted values from the model")
print("*"*50)
print(r2_score(y_test,y_pred)) # Best value is 1.0

print("*"*50)
print("Calculate the cross validation score 10 fold with default initilaization of KNN model ie 5 nearest neighbour and distance metric of miskowski")
print("*"*50)
knn = KNeighborsRegressor()
print(cross_val_score(knn,X,y,cv=10) )# 10-fold cross validation default is 3


print("*"*50)
print("Calculate the cross validation score 10 fold with neighbour ranging from 1 ..10 and different distance metrics and find the best estimator")
print("*"*50)
params = {"n_neighbors":np.arange(1,10,1), "metric":["euclidean", "minkowski","manhattan", "jaccard", "cosine"]}
knn = KNeighborsRegressor()
knn_cv = GridSearchCV(knn,params,cv=10)
knn_cv.fit(X,y)
print(knn_cv.best_score_)
print(knn_cv.best_estimator_)