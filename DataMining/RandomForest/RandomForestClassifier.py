import pandas as pd
import numpy as np
import warnings

from sklearn.ensemble import VotingClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split as tts, GridSearchCV , RandomizedSearchCV

print("*"*50)
print("Read Csv file and check for missing values")
print("*"*50)
df = pd.read_csv("gapminder.csv")
print(df.isnull().sum())

print("*"*50)
print("Reading in the dataset and stored as df, Keeping a copy of original df as df_original")
print("*"*50)
df_original  = df.copy()
print(df.head())
print(df.shape)

print("*"*50)
print("Defining X and y. Basically you want to get predict Region from the set of values")
print("*"*50)
X = df.drop(["Region"],1)
y = df["Region"]

print("*"*50)
print("Creating different models to run the voting classifier")
print("*"*50)
model1 = KNeighborsClassifier()

print("*"*50)
print("Using GridSearch CV to the best estimaor for KNN classifier")
print("*"*50)
params = {"n_neighbors":np.arange(2,10),"metric":["cosine", "minkowski", "euclidean", "manhattan", "jaccard"]}
model1_cv = GridSearchCV(estimator=model1, param_grid=params, cv = 5)
model1_cv.fit(X,y)
print(model1_cv.best_score_)
print(model1_cv.best_params_)
model1= model1_cv.best_estimator_
print(model1)

print("*"*50)
print("Using GridSearch CV to the best estimaor for Decision tree")
print("*"*50)
model2 = DecisionTreeClassifier(random_state=42)
params = {"criterion":["gini", "entropy"], "max_depth":np.arange(2,8),"min_samples_split":np.arange(0.01, 0.1, 0.01)}
model2_cv = GridSearchCV(estimator=model2, param_grid = params, cv = 5)
model2_cv.fit(X,y)
print(model2_cv.best_score_)
print(model2_cv.best_params_)
model2= model2_cv.best_estimator_
print(model2)
print(model2.feature_importances_)

print("*"*50)
print("Dividing the dataset into train and test")
print("*"*50)
X_train, X_test, y_train, y_test = tts(X,y, test_size = 0.25, random_state = 42, stratify = y)

print("*"*50)
print("Method to generate accuracy score and classification report for different models")
print("*"*50)
def generate_classfication_report(model):
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    print (classification_report(y_test,y_pred))
    return accuracy_score(y_test,y_pred)


print("*"*50)
print("Using Voting classifier to get best result through majority voting")
print("*"*50)
model = VotingClassifier(estimators=[('knn', model1), ('dtc', model2)], voting='soft') #Since there 2 models Soft voting works best because of probability.
generate_classfication_report(model)

print("*"*50)
print("Using Bagging classifier to get model accuracy on different datasetup")
print("*"*50)
bc = BaggingClassifier(base_estimator=model2, random_state=42, oob_score=True, n_estimators=40, max_samples=0.72) #Base estimator has been used as Decision tree
generate_classfication_report(bc)

bc.fit(X_train,y_train)
y_pred = bc.predict(X_test)
print(bc.oob_score_)

print("*"*50)
print("Executing Gridsearch CV on the Bagging classifier")
print("*"*50)
params ={"max_samples":np.arange(0.7,1,0.1), "n_estimators":np.arange(10,50,10)}
bc_cv = GridSearchCV(estimator=bc, param_grid=params, cv = 5)
bc_cv.fit(X,y)
print(bc_cv.best_params_)
print(bc_cv.best_score_)
print(bc_cv.best_estimator_)

print("*"*50)
print("Executing Random Forest classifier")
print("*"*50)
rfc = RandomForestClassifier(random_state=42, criterion = "entropy", max_depth=6, min_samples_split=0.12, oob_score = True, n_estimators = 40, class_weight = "balanced")
rfc.fit(X_train,y_train)
y_pred = rfc.predict(X_test)
rfc.score(X_test,y_test)
print (classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print(rfc.feature_importances_)