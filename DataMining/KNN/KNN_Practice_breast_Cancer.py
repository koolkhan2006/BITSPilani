import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from matplotlib import pyplot as plt
import seaborn as sns


print("*"*50)
print("Read Csv file")
print("*"*50)
df = pd.read_csv("breast_cancer.csv")
print(df)

print("*"*50)
print("Work on first 10 rows")
print("*"*50)
print(df.head())

print("*"*50)
print("Before working on numerical values ensure there is no null or NAN value")
print("*"*50)
print(df.columns)
print("**" * 50)
print(df.info())

df.drop(columns=[df.columns[0],df.columns[-1]],axis=1,inplace=True)
print(df.info())

df.diagnosis = df.diagnosis.replace({'M':1,'B':0})
plt.figure(figsize=(10,10))
print(df.iloc[:,0:11])

sns.heatmap(df.iloc[:,0:11].corr(),annot=False, cmap="coolwarm")

df.head()

Y = df.iloc[0:,0]
print(Y)

X = df.iloc[0:,1:11]
print(X)
# removing "texture_mean" since
X.drop(columns=[df.columns[2]],axis=1,inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X,Y,random_state = 50, test_size = 0.35)
knn = KNeighborsClassifier(n_neighbors=3,metric='euclidean')
knn.fit(X_train, y_train)
print(knn.score(X_test,y_test))

print(cross_val_score(knn,X,Y,cv=10))
knn = KNeighborsClassifier()
params = {"n_neighbors":np.arange(1,10,1), "metric":["euclidean", "minkowski", "jaccard", "cosine"]}
knn_cv = GridSearchCV(estimator=knn, param_grid=params, cv = 10)
knn_cv.fit(X,Y)

print(knn_cv.best_params_)

print(knn_cv.best_estimator_)