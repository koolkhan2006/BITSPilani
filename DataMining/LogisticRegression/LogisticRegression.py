import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV, RFE
from sklearn.model_selection import train_test_split as tts, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("*"*50)
print("Read Csv file")
print("*"*50)
df = pd.read_csv("diabetes.csv")
print(df.isnull().sum())

print("*"*50)
print("Reading in the dataset and stored as df, Keeping a copy of original df as df_original")
print("*"*50)
df_original  = df.copy()
print(df.head())
print(df.shape)

print("*"*50)
print("Definen X and y")
print("*"*50)
X = df.drop(["Outcome"],1)
y = df["Outcome"]

print("*"*50)
print("Dividing the dataset into train and test")
print("*"*50)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state = 42)

print("*"*50)
print("Because this involves working with distance based ML algorithms, "
      "Scaling will be done only for the numerical data.")
print("*"*50)
scaler = StandardScaler()
X_scale = pd.DataFrame(scaler.fit_transform(X_train), columns= list(df.iloc[:,:-1]))
X_test_scale = pd.DataFrame(scaler.fit_transform(X_test), columns= list(df.iloc[:,:-1]))
print(X_scale)

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
X_new = X[["DiabetesPedigreeFunction", "Pregnancies"]]
log_regcv = LogisticRegressionCV()
log_regcv.fit(X_new,y)
print(log_regcv.score(X_new,y))

print("*"*50)
print("Statistical model for sm to check for P value(Wald test) and to get the Mc faddens Pseudo R2 score")
print("*"*50)
X_sm = sm.add_constant(X)
model = sm.Logit(y,X_sm).fit()
drop = list(X_sm)
drop.remove("Age")
X_new = X_sm[drop]
model = sm.Logit(y,X_new).fit()
model.summary()

print("*"*50)
print("Feature selection using the Recursive feature elimination")
print("*"*50)
f2 = ["Pregnancies", "Glucose", "BMI", "DiabetesPedigreeFunction"]
log_reg = LogisticRegression()
rfe_cv = RFECV(estimator=log_reg, cv=10)
rfe = RFE(estimator=log_reg, n_features_to_select=4)
rfe_cv.fit(X,y)
print(rfe_cv.ranking_)
print (list(X))

f2 = ["Pregnancies", "Glucose", "BMI", "DiabetesPedigreeFunction"]
X = X[f2]


log_reg = LogisticRegression(random_state=42)
X_train, X_test, y_train, y_test = tts(X,y,test_size = 0.25, random_state = 42)
model = log_reg.fit(X_train,y_train)
y_pred = model.predict(X_test)
accuracy_score(y_test,y_pred)
print (classification_report(y_test,y_pred))
print (confusion_matrix(y_test,y_pred))