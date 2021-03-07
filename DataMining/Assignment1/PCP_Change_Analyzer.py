import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split ,GridSearchCV
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFECV, RFE
import numpy as np
import matplotlib.pyplot as plt

print("*"*50)
print("Read Csv file")
print("*"*50)
df = pd.read_csv("DataSet_PCP_Change.csv")
print(df.head())
print(df.shape)

print("*"*50)
print("Reading in the dataset and stored as df, Keeping a copy of original df as df_original")
print("*"*50)
df_original  = df.copy()
print(df_original.head())
print(df_original.shape)

print("*"*50)
print("Check count for missing values in each column")
print("*"*50)
print(df_original.isnull().sum())

print("*"*50)
print("Dropping test index as it is just an index column")
print("*"*50)
df_original = df_original.drop(['testindex'], axis=1)
print(df_original.head())

print("*"*50)
print("Dropping claims_daysaway as most of the columns are empty")
print("*"*50)
df_original = df_original.drop(['claims_daysaway'], axis=1)
print(df_original.head())

print("*"*50)
print("Dropping missing values in tier as it is ver less in numbers")
print("*"*50)
df_original = df_original.dropna()
print(df_original.head())

print("*"*50)
print("Create a dataset and remove the last column")
print("*"*50)
X = df_original.drop(['outcome'], axis=1)
y =  df_original['outcome']
print(X.head())
print(y.head())

print(y.value_counts())

print("*"*50)
print("Check count for missing values in each column")
print("*"*50)
print(X.isnull().sum())

print("*"*50)
print("Dividing the dataset into train and test")
print("*"*50)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state = 42)

print("*"*50)
print("Because this involves working with distance based ML algorithms, "
      "Scaling will be done only for the numerical data.")
print("*"*50)
scaler = StandardScaler()
X_scale = pd.DataFrame(scaler.fit_transform(X_train), columns= list(X_train))
X_test_scale = pd.DataFrame(scaler.fit_transform(X_test), columns= list(X_test))
print(X_scale.shape)
print(X_test_scale.shape)

print("*"*50)
print("Apply Logistic Regression")
print("*"*50)
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X,y)
# y_pred = log_reg.predict(X_test_scale)
print(log_reg.score(X,y))

print("*"*50)
print("Apply Logistic Regression Cross validation")
print("*"*50)
log_regcv = LogisticRegressionCV()
log_regcv.fit(X,y)
print(log_regcv.score(X,y))

print("*"*50)
print("Find the feature importance using the model coefficients")
print("*"*50)
features = list(X)
feature_weights = np.abs(log_regcv.coef_).tolist()[0]
d = dict(zip(features, feature_weights))
d = pd.DataFrame(list(d.items()), columns=["features", "ranking"])
d = d.sort_values(["ranking"], ascending=False)
print(d)

print("*"*50)
print("After calculating the feature the importance using the model itself then feed in the most important features and check the score.")
print("*"*50)
X_new = X[["pcp_lookback", "same_address","kid","is_ped"]]
log_regcv = LogisticRegressionCV()
log_regcv.fit(X_new,y)
print(log_regcv.score(X_new,y))

# print("*"*50)
# print("Feature selection using the Recursive feature elimination")
# print("*"*50)
# f2 = ["pcp_lookback", "same_address","kid","is_ped"]
# log_reg = LogisticRegression()
# rfe_cv = RFECV(estimator=log_reg, cv=5)
# rfe_cv.fit(X,y)
# print(rfe_cv.ranking_)
# print (list(X))

log_reg = LogisticRegression(random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25, random_state = 42)
model = log_reg.fit(X_train,y_train)
y_pred = model.predict(X_test)
accuracy_score(y_test,y_pred)
print (classification_report(y_test,y_pred))
print (confusion_matrix(y_test,y_pred))

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
print (classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print(model2.feature_importances_)

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
model = VotingClassifier(estimators=[ ('dtc', model2),('rfc',rfc)], voting='soft') #Since there 2 models Soft voting works best because of probability.
print(generate_classfication_report(model))

print("*"*50)
print("Using Bagging classifier to get model accuracy on different datasetup")
print("*"*50)
bc = BaggingClassifier(base_estimator=model2, random_state=42, oob_score=True, n_estimators=40, max_samples=0.72) #Base estimator has been used as Decision tree
generate_classfication_report(bc)

bc.fit(X_train,y_train)
y_pred = bc.predict(X_test)
print(bc.oob_score_)
