import numpy as np
import pandas as pd
import time

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import stats
def standard_err(y_true,y_pred):
    gradient, intercept, r_value, p_value, std_err = stats.linregress(y_true,y_pred)
    return std_err
start1 = time.time()
print(start1)
print("*"*50)
print("Read Csv file")
print("*"*50)
df = pd.read_csv("gapminder.csv")
print(df.head())
print(df.shape)

print("*"*50)
print("Model is Linear Regression")
print("*"*50)
model = "Linear Regression"

print("*"*50)
print("Reading in the dataset and stored as df, Keeping a copy of original df as df_original")
print("*"*50)
df_original  = df.copy()
print(df_original.head())
print(df_original.shape)

print("*"*50)
print("Because this involves working with distance based ML algorithms, "
      "we will scale/normalize numerical variables and one hot encode categorical variables"
      "Scaling will be done only for the numerical data and hence Region column is removed.")
print("*"*50)
scaler = StandardScaler()
num = pd.DataFrame(scaler.fit_transform(df.iloc[:,:-1]), columns=list(df.iloc[:,:-1]))
df = pd.concat([num, pd.DataFrame(df["Region"])],1)
print(df.head())

print("*"*50)
print("Performing One Hot encoding on the categorical data")
print("*"*50)
df = pd.get_dummies(df)
print(df.head())

print("*"*50)
print("Dropping one of the categorical column. Minimum feature you give to the model it is the best")
print("*"*50)
df.drop(["Region_South Asia"],1,inplace = True)
print(df.head())

print("*"*50)
print("Defining X and y. Basically you want to get predict life from the rest of the features")
print("*"*50)
X = df.drop(["life"],1)
y = df["life"]

print(X.head())
print(y.head())

print("*"*50)
print("Importing Linear Regression")
print("*"*50)
lin_reg  = LinearRegression()

print("*"*50)
print("Manual train test split")
print("*"*50)
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 42, test_size = 0.25)

print("*"*50)
print("First run of manual train and test split for linear regression model")
print("*"*50)
lin_reg.fit(X_train,y_train)
y_pred = lin_reg.predict(X_test)

print("*"*50)
print("Calculate the R2 score ")
print("*"*50)
print ("R squared "+str(model)+" "+str(r2_score(y_test,y_pred)))
print ("RMSE squared " + str(model) +" " + str(np.sqrt(mean_squared_error(y_test, y_pred))))
print ("Standard error "+str(model)+" "+str(standard_err(y_test,y_pred)))

print("*"*50)
print("There are no hyperparameters to control in Linear Regression. "
       "That comes with regularization. "
      "Seeing the features the line without any feature selection has taken")
print("*"*50)
features = list(X)
feature_weights = np.abs(lin_reg.coef_).tolist()
d = dict(zip(features, feature_weights))
d = pd.DataFrame(list(d.items()), columns=["features", "ranking"])
d = d.sort_values(["ranking"], ascending=False)
print(d)

print("*"*50)
print ("Top 5 features chosen by linear regression are:")
print("*"*50)
time.sleep(3)
f1 = d[:5]["features"].tolist()
print ("\n")
print (f1)