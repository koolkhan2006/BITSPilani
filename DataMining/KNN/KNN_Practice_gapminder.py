from scipy.spatial.distance import euclidean
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import r2_score, accuracy_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("gapminder.csv")
print(df.head())
print(df.shape)
df1 = df.copy()
df.drop(['Region'],1,inplace=True)
print(df.head())
df = df[["fertility", "life", "child_mortality"]]
df = df[:10]
print(df)
point_0 = df.iloc[0][:2]
print(point_0)
point_1 = df.iloc[1][:2]
print(point_1)

print(euclidean(point_0,point_1))
print(euclidean(point_0.tolist(),point_1.tolist()))

poi = [1.62, 75.6]

distances = []
for x in range(df.shape[0]):
    point_x = df.iloc[x][:2]
    distances.append(euclidean(poi,point_x.tolist()))

df["distances"]= distances
print(df)
print(df.sort_values(["distances"]))

print((30.8+29.5)/3)
df.drop(['distances'],1,inplace=True)
print(df)

X =  df.drop(['child_mortality'],1)
Y =  df['child_mortality']

print(X)
print(Y)

knn = KNeighborsRegressor(n_neighbors=3, metric="euclidean")
knn.fit(X,Y)

poi = pd.DataFrame(poi).T
X_test = poi
X_test.columns = ["fertility", "life"]
print(X_test)

Y_pred = knn.predict(X_test)
print(Y_pred)

X_train, X_test, y_train, y_test = tts(X,Y, test_size = 0.25, random_state = 42)

print(y_test)
knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)

print(y_pred)
print(r2_score(y_test, y_pred))

knn = KNeighborsRegressor(n_neighbors=3)
min(cross_val_score(knn, X,Y, cv = 10))

params = {"n_neighbors":[2,3,4,5,6,7],
         "metric":["euclidean", "cosine", "jaccard", "manhattan", "minkowski"]}

knn = KNeighborsRegressor()
knn_cv = GridSearchCV(estimator=knn, param_grid=params, cv = 10)
knn_cv.fit(X,Y)

print(knn_cv.best_estimator_)