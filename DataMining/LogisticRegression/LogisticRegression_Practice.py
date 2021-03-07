import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import r2_score ,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("*"*50)
print("Read Csv file")
print("*"*50)
df = pd.read_csv("Social_Network_Ads.csv")
print(df.isnull().sum())

# Since none of the columns have null / Nan / missing values there is no need of imputations to fill in the missing values.

print("*"*50)
print("Reading in the dataset and stored as df, Keeping a copy of original df as df_original")
print("*"*50)
df_original  = df.copy()
print(df.head())
print(df.shape)

print("*"*50)
print("Definen X and y")
print("*"*50)
X = df.drop(["Purchased","User ID","Gender","EstimatedSalary"],1)
y = df["Purchased"]

print(X.head())
print(y.head())

print("*"*50)
print("Dividing the dataset into train and test")
print("*"*50)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)

print("*"*50)
print("Because this involves working with distance based ML algorithms, "
      "Scaling will be done only for the numerical data.")
print("*"*50)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns= list(X))
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns= list(X))
print(X_scaled)

print("*"*50)
print("Apply Logistic Regression")
print("*"*50)
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_scaled,y_train)
y_pred = log_reg.predict(X_test_scaled)
# print(log_reg.score(X,y))
cm = confusion_matrix(y_test,y_pred)
print(cm)

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
# As per the analysis of the coefficients it is proved that role of salary is negligible in terms of purchase decision of the individual.

print("*"*50)
print("Calculate the Mc faddens Pseudo R2 score for Logistic Model")
print("*"*50)

print("*"*50)
print("Statistical model for sm to check for P value(Wald test) and to get the Mc faddens Pseudo R2 score")
print("*"*50)



