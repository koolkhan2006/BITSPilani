import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


lst = [1,2,3,4,5,1,3,2,1]
numpy_array = np.array(lst)
panda_series = pd.Series(lst)
print(panda_series.sort_values().mean())
print(panda_series.sort_values().median())



print("#"*20)
print("Box Plot")
print("#"*20)
lst = [1,1,2,3,4,5,6,7,8,9,10,16]
panda_series = pd.Series(lst)
print(panda_series.sort_values().mean())
print(panda_series.sort_values().median())
print(panda_series.sort_values().quantile(0.75, interpolation='midpoint'))
plt.boxplot(lst)

#plt.show()

print("#"*20)
print("Read train.csv data")
print("#"*20)
data = pd.read_csv("train.csv") # read.csv() is used for rendering a csv in Pandas
print(data.head())# Recollect .head() returns top 5 rows.
Sales_price = pd.Series(data['SalePrice'])
print(Sales_price.head(15))


print("#"*20)
print("Range of sale price")
print("#"*20)
print(Sales_price.min())
print(Sales_price.max())
print(Sales_price.max() - Sales_price.min())
greater_than_1_lakh  = data['SalePrice'] >= 100000
lessthan_than_1_lakh_50000  = data['SalePrice'] <= 150000
df = data[greater_than_1_lakh & lessthan_than_1_lakh_50000 ]
print(data['Id'].nunique())
print(df['Id'].nunique())


print("#"*20)
print("Mean value of housing data")
print("#"*20)
mean = np.mean(Sales_price)
print(mean)

print("#"*20)
print("Median value of housing data")
print("#"*20)
median = np.median(Sales_price)
print(median)

print("#"*20)
print("Quantile value of housing data with default interpolation")
print("#"*20)
q1 = Sales_price.quantile(0.25) # upper quartile  } Note: The fuction is .quantile() with 'n'
q3 = Sales_price.quantile(0.75) # lower quartile  }       not .quartile() with 'r'
print("Q1:", q1)
print("Q3:", q3)
print("IQR:", q3 - q1)

print("#"*20)
print("Quantile value of housing data with midpoint interpolation")
print("#"*20)
q1 = Sales_price.quantile(0.25,interpolation='midpoint') # upper quartile  } Note: The fuction is .quantile() with 'n'
q3 = Sales_price.quantile(0.75,interpolation='midpoint') # lower quartile  }       not .quartile() with 'r'
print("Q1:", q1)
print("Q3:", q3)
print("IQR:", q3 - q1)

print("#"*20)
print("Quantile value of housing data with higher interpolation")
print("#"*20)
q1 = Sales_price.quantile(0.25,interpolation='higher') # upper quartile  } Note: The fuction is .quantile() with 'n'
q3 = Sales_price.quantile(0.75,interpolation='higher') # lower quartile  }       not .quartile() with 'r'
print("Q1:", q1)
print("Q3:", q3)
print("IQR:", q3 - q1)

print("#"*20)
print("Quantile value of housing data with lower interpolation")
print("#"*20)
q1 = Sales_price.quantile(0.25,interpolation='lower') # upper quartile  } Note: The fuction is .quantile() with 'n'
q3 = Sales_price.quantile(0.75,interpolation='lower') # lower quartile  }       not .quartile() with 'r'
print("Q1:", q1)
print("Q3:", q3)
print("IQR:", q3 - q1)

print("#"*20)
print("Quantile value of housing data with linear interpolation")
print("#"*20)
q1 = Sales_price.quantile(0.25,interpolation='linear') # upper quartile  } Note: The fuction is .quantile() with 'n'
q3 = Sales_price.quantile(0.75,interpolation='linear') # lower quartile  }       not .quartile() with 'r'
print("Q1:", q1)
print("Q3:", q3)
print("IQR:", q3 - q1)

#plt.boxplot(Sales_price)
#plt.show()

print("#"*20)
print("Find no. of outliers")
print("#"*20)
outlier_lower_limit = q1 - 1.5*(q3 - q1)
outlier_upper_limit = q3 + 1.5*(q3 - q1)
print(outlier_lower_limit)
print(outlier_upper_limit)
lower_limit_outliers = Sales_price[Sales_price < outlier_lower_limit].count()
upper_limit_outliers = Sales_price[Sales_price > outlier_upper_limit].count()
print("lower_limit_outliers:", lower_limit_outliers)
print("upper_limit_outliers:", upper_limit_outliers)
print("total outliers:", upper_limit_outliers + lower_limit_outliers)

plt.hist(Sales_price, bins=60)
#plt.show()

print("#"*20)
print("Standard deviation")
print("#"*20)
lst = [42,48,45,40,60,38]
numpy_array = np.array(lst)
panda_series = pd.Series(lst)
print("Mean ",panda_series.sort_values().mean())
print("Median ", panda_series.sort_values().median())
print("Mode ", panda_series.sort_values().mode())
print("Variance ", panda_series.sort_values().var())
print("Square root of variance ", np.sqrt(panda_series.sort_values().var()))
print("Standard deviation ", panda_series.sort_values().std())
print("Mean Absolute deviation ", panda_series.sort_values().mad())

print("#"*20)
print("Standard Deviation for Lot Area")
print("#"*20)
LotArea_price = pd.Series(data['LotArea'])
print("Mean ",LotArea_price.sort_values().mean())
print("Median ", LotArea_price.sort_values().median())
print("Mode ", LotArea_price.sort_values().mode())
print("Variance ", LotArea_price.sort_values().var())
print("Square root of variance ", np.sqrt(LotArea_price.sort_values().var()))
print("Standard deviation ", LotArea_price.sort_values().std())
print("Mean Absolute deviation ", LotArea_price.sort_values().mad())

print("#"*20)
print("Standard Deviation for Lot Area using Numpy")
print("#"*20)
LotArea_price = np.sort(pd.Series(data['LotArea']).to_numpy())
print("Mean ",LotArea_price.mean())
print("Variance ",np.var(LotArea_price))
print("Standard Deviation from variance ",np.sqrt(np.var(LotArea_price)))
print("Standard Deviation ",np.std(LotArea_price))



