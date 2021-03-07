import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, accuracy_score, classification_report, confusion_matrix

print("*"*50)
print("Read Csv file")
print("*"*50)
df = pd.read_csv("gapminder.csv")

print("*"*50)
print("Work on first 10 rows")
print("*"*50)
df1 =  df.copy()
df = df[:10]
print(df)

print("*"*50)
print("Work only on specific columns")
print("*"*50)
df =  df[['fertility','life','child_mortality']]
print(df)

print("*"*50)
print("calculate the euclidean distance with manual approach")
print("*"*50)
point_0 = df.iloc[0][:2]
point_1 = df.iloc[1][:2]
print(euclidean(point_0,point_1))

print("*"*50)
print("calculate the euclidean distance with manual approach for point of interest")
print("*"*50)
poi = [1.62, 75.6]
distances = []
for x in range(df.shape[0]):
    point_x = df.iloc[x][:2]
    distances.append(euclidean(poi,point_x))
df['distances'] = distances
print(df.sort_values('distances'))
print(np.mean([15.4,15.4,29.5]).ravel())

print("*"*50)
print("Predict the value for Poi using the KNNRegressor fit predict method")
print("*"*50)
X = df.drop(['distances','child_mortality'],1)
Y = df['child_mortality']
print(X)
print(Y)
knn = KNeighborsRegressor(n_neighbors=3,metric="euclidean")
knn.fit(X,Y)
x_test = pd.DataFrame(poi).T
x_test.columns = ['fertility','life']
print(x_test)
y_pred = knn.predict(x_test)
print(y_pred)

print("*"*50)
print("Check accuracy of your model using the train test split with fixed random state of 42"
      "and then checking the accuracy score using r2")
print("*"*50)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.25, random_state = 42)
knn.fit(X_train,Y_train)
y_pred = knn.predict(X_test)
print(y_pred)
print(r2_score(Y_test,y_pred))

print("*"*50)
print("Check accuracy of your model using cross validation score")
print("*"*50)
knn = KNeighborsRegressor(n_neighbors=2)
print(min(cross_val_score(knn,X,Y,cv=5)))

print("*"*50)
print("Check accuracy of your model using the Grid Search CV which takes possible combinations of hyper parameters")
print("*"*50)
params = {"n_neighbors":[2,3,4,5,6,7],
         "metric":["euclidean", "cosine", "jaccard", "manhattan", "minkowski"]}
knn = KNeighborsRegressor()
knn_cv = GridSearchCV(estimator=knn, param_grid=params, cv = 10)
knn_cv.fit(X,Y)
print(knn_cv.best_estimator_)