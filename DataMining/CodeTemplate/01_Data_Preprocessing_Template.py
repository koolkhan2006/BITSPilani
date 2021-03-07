import pandas as pd
import warnings

from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import Imputer, LabelEncoder , OneHotEncoder

print("*"*50)
print("Read Csv file")
print("*"*50)
df = pd.read_csv("Data.csv")

print("*"*50)
print("Check count for missing values in each column")
print("*"*50)
print(df.isnull().sum())

print("*"*50)
print("Create a dataset and remove the last column")
print("*"*50)
X = df.iloc[:,:-1].values
y =  df.iloc[:, 3].values
print(X)
print(y)

print("*"*50)
print("Use Imputer to fill in the missing data")
print("*"*50)
imputer  = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer =  imputer.fit(X[:,1:3])
X[:,1:3] =  imputer.transform(X[:,1:3])
print(X)

print("*"*50)
print("Label encoding the first column")
print("*"*50)
labelencoder_X = LabelEncoder()
labelencoder_y = LabelEncoder()
# X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
y =  labelencoder_y.fit_transform(y)
# print(X)
print(y)

print("*"*50)
print("One Hot Encoding first column")
print("*"*50)
# oneHotEncoder_X = OneHotEncoder(categorical_features= [0])
X = pd.DataFrame(X,columns=list(df.iloc[:,:-1]))
X = pd.get_dummies(X,drop_first='true')
print(X)

print("*"*50)
print("Train test Split on this data")
print("*"*50)
lin_reg = LinearRegression()
X_train, X_test , y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=21)
lin_reg.fit(X,y)
# y_pred = lin_reg.predict(X_test)
print(lin_reg.score(X,y))