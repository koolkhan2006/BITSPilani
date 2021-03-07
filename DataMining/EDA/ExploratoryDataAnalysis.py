import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from scipy import stats
import statsmodels.api as sm
def standard_err(y_true,y_pred):
    gradient, intercept, r_value, p_value, std_err = stats.linregress(y_true,y_pred)
    return std_err
print("*"*50)
print("Read Csv file")
print("*"*50)
df = pd.read_csv("Automobile_data.csv")
print(df.head())
print(df.shape)

print("*"*50)
print("Reading in the dataset and stored as df, Keeping a copy of original df as df_original")
print("*"*50)
df_original  = df.copy()
print(df_original.head())
print(df_original.shape)

print("*"*50)
print("Find the object type of the normalized losses")
print("*"*50)
print(df['normalized-losses'].dtype)

print("*"*50)
print("Find the object type of the normalized losses")
print("*"*50)
def replace_clean(df, column_name, punctuation_symbol):
    m = df[(df[column_name]!=punctuation_symbol)][column_name].astype(float).mean()
    df[column_name] = df[column_name].replace(punctuation_symbol, m)
    df[column_name] = df[column_name].astype(float)
    return df

print("*"*50)
print("Converting normalized the losses")
print("*"*50)
replace_clean(df,'normalized-losses','?')
print(df['normalized-losses'].dtype)

print("*"*50)
print("get the list of columns which have questions marks")
print("*"*50)
cols = df.isin(['?']).any()[df.isin(['?']).any()==True].index.tolist()
print(cols)
cols.remove('num-of-doors')
print(cols)

print("*"*50)
print("Loop over list of columns and replace question mark with mean value")
print("*"*50)
for x in cols:
     replace_clean(df,x,"?")
cols = df.isin(['?']).any()[df.isin(['?']).any()==True].index.tolist()
print(cols)

print("*"*50)
print("Since number of doors is categorical we need to replace by mode and NOT mean")
print("*"*50)
m = df[(df["num-of-doors"]!="?")]["num-of-doors"].mode().tolist()[0]
df["num-of-doors"] = df["num-of-doors"].replace("?", m)
cols = df.isin(['?']).any()[df.isin(['?']).any()==True].index.tolist()
print(cols)
df_clean = df.copy()
print(df_clean.head())

print("*"*50)
print("Create a dataframe with only numerical and categorical value")
print("*"*50)
X =  df_clean.drop(['price'],1)
Y =  df_clean['price']
X_number = X.select_dtypes(include = np.number)
X_Category = X.select_dtypes(exclude = np.number)
print(X_number.head())
print(X_Category.head())
scaler = StandardScaler()
X_number = pd.DataFrame(scaler.fit_transform(X_number), columns=list(X_number))
X_Category = pd.get_dummies(X_Category, drop_first=True)
print(X_Category.shape)
X= pd.concat([X_number,X_Category],1)
print(X.shape)

print("*"*50)
print("Apply KNN  ")
print("*"*50)
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.25, random_state = 42)
print(y_test)
knn = KNeighborsRegressor()
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
print(y_pred)
print(r2_score(y_test, y_pred))

print("*"*50)
print("Univariate analysis")
print("*"*50)
# df_clean.make.value_counts().nlargest(10).plot(kind='bar', figsize=(15,5))
plt.title("Number of vehicles by make")
plt.ylabel('Number of vehicles')
plt.xlabel('Make')
# plt.show()

print("*"*50)
print("Scatter plot to check whethere data is linear or not")
print("*"*50)
x = np.arange(0,205,1)
# plt.scatter(Y,x)
# plt.show()

print("*"*50)
print("Histogram to check whethere data is linear or not")
print("*"*50)
plt.hist(Y)
plt.show()


print("*"*50)
print("Apply Linear Regression")
print("*"*50)
model = "Linear Regression"
lin_reg  = LinearRegression()
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
print("Apply Stats model")
print("*"*50)
X = sm.add_constant(X)
model = sm.OLS(Y,X).fit()
print(model.summary())
results_summary = model.summary()
results_as_html = results_summary.tables[1].as_html()
pval = pd.read_html(results_as_html, header=0, index_col=0)[0]
print(pval['P>|t|'][pval['P>|t|']<=0.05])