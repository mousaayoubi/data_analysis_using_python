import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

#Specify url to import
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"

#Specify header titles
headers = ["symboling", "normalized-losses", "make", "fuel-type", "aspiration", "num-of-doors", "body-style", "drive-wheels", "engine-location", "wheel-base", "length", "width", "height", "curb-weight", "engine-type", "num-of-cylinders", "engine-size", "fuel-system", "bore", "stroke", "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg", "price"]

#Read from a csv file
df = pd.read_csv(url, header = None)

#Assign headers
df.columns = headers

#Print first 10 headers
print(df.head(10))

#Specify path to export new dataset
path = "C://Program Files/Git/Projects/datascience/new_dataset.csv"

#Export new dataset to new file
df.to_csv(path)

#Show datatypes
print(df.dtypes)

#Show description of the data
print(df.describe(include="all"))

#Show summary of the data
print(df.info())

#Change ? to NaN
df.replace("?", np.nan, inplace=True)
print(df)

#Change normalized-losses column to fload data type
df["normalized-losses"] = df["normalized-losses"].astype("float")
print(df.dtypes)

#Drop na values from the price column
df.dropna(subset=["price"], axis=0, inplace=True)
print(df)

#Replace empty normalized-losses column with the average value
mean = df["normalized-losses"].mean()
df["normalized-losses"].replace(np.nan, mean, inplace=True)
print(df)

#Change fuel efficiency from mpg to L/100 km
df["city-mpg"] = 235/df["city-mpg"]
df.rename(columns={"city-mpg": "city-L/100km"}, inplace=True)
print(df["fuel-type"])

#Get dummy variable counts for fuel-type column
df2 = pd.get_dummies(df["fuel-type"])
print(df2)

#Concatunate both dataframes to show new dummy variable counts
df = pd.concat([df, df2], axis=1)
print(df)

#Count number of values in drive-wheels column
print(df["drive-wheels"].value_counts())

df["price"] = df["price"].astype("int64")
df.dropna(subset=["horsepower"], axis=0, inplace=True)
df["horsepower"] = df["horsepower"].astype("int64")
print(df.dtypes)

#Show relationship between engine-size and price using a scatter plot
x = df["engine-size"]
y = df["price"]
plt.scatter(x, y)
plt.xlabel("Engine Price")
plt.ylabel("Price")
plt.title("Relationship Between Engine Size and Price")
plt.show()

#Show Relationship between drive-wheel, body-style and price using a heatmap
df4 = df[["drive-wheels", "body-style", "price"]]
df_grp = df4.groupby(["drive-wheels", "body-style"], as_index=False).mean()
print(df_grp)

df_pivot = df_grp.pivot(index="drive-wheels", columns="body-style")
print(df_pivot)

plt.pcolor(df_pivot, cmap="RdBu")
plt.colorbar()
plt.xlabel("Body Style")
plt.ylabel("Type of Drive Wheel")
plt.title("Relationship Between Body Style, Type of Drive Wheel and Price")
ax = plt.subplot()
ax.set_xticklabels(["Convertibe", "Hardtop", "Hatchback", "Sedan", "Wagon"], horizontalalignment="center")
ax.set_yticklabels([" ", "4WD", " ", "FWD", " ", "RWD"], verticalalignment= "center")
plt.show()

#Use ANOVA to check correlation and significance between grouped set of data such as make and price
df_anova = df[["make", "price"]]
print(df_anova)
anova_grp = df_anova.groupby(["make"])

#Compare ANOVA for variance in price between honda and subaru car makers
anova_results = stats.f_oneway(anova_grp.get_group("honda")["price"], anova_grp.get_group("subaru")["price"])
print(anova_results)

#Compare ANOVA for variance in price between honda and jaguar car makers
anova_results = stats.f_oneway(anova_grp.get_group("honda")["price"], anova_grp.get_group("jaguar")["price"])
print(anova_results)

#Plot correlation between car engine size and price
x = df["engine-size"]
y = df["price"]

sns.regplot(x="engine-size", y="price", data=df)
plt.ylim(0,)
plt.title("Correlation between car engine size and price")
plt.show()

#Show the residual plot to check if the linear model is appropriate for engine-size and price
sns.residplot(x, y)
plt.ylim(-15000,)
plt.title("Residual plot of regression linear model of engine-size and price")
plt.show()

#Plot correlation between car highway mpg and price
x = df["highway-mpg"]
y = df["price"]

sns.regplot(x="highway-mpg", y="price", data=df)
plt.ylim(0,)
plt.title("Correlation between car highway mpg and price")
plt.show()

#Show the residual plot to check if the linear model is appropriate for car highway mpg and price
sns.residplot(x, y)
plt.ylim(-20000,)
plt.title("Residual plot of regression linear model of car highway mpg and price")
plt.show()

#Plot correlation between car peak rpm and price
df["peak-rpm"] = df["peak-rpm"].astype("float")
x = df["peak-rpm"]
y = df["price"]

sns.regplot(x="peak-rpm", y="price", data=df)
plt.ylim(0,)
plt.title("Correlation between car peak rpm and price")
plt.show()

#Show the residual plot to check if the linear model is appropriate for car peak rpm and price
sns.residplot(x, y)
plt.ylim(-30000,)
plt.title("Residual plot of regression linear model of car peak rpm and price")
plt.show()

#Fit the model using polynomial regression with 1 dimension
f = np.polyfit(x, y, 3)
p = np.poly1d(f)
print(f)

#Fit the model using polynomial regression with multiple dimensions
pr=PolynomialFeatures(degree=2)
x_polly = pr.fit_transform(df[["horsepower", "peak-rpm"]])
print(x_polly)

#Normalize features using StandardScaler
SCALE = StandardScaler()
SCALE.fit(df[["horsepower", "highway-mpg"]])
x_scale = SCALE.transform(df[["horsepower", "highway-mpg"]])
print(x_scale)

#Show pearson coefficient and p-value for variables horsepower and price
df["horsepower"] = df["horsepower"].astype("float")
df.dropna(subset=["horsepower"], axis=0, inplace=True)
print(df["horsepower"])

pearson_coef, p_value = stats.pearsonr(df["horsepower"], df["price"])
print(pearson_coef)
print(p_value)

#Show pearson coefficient and p-value for variables highway mpg and price
pearson_coef, p_value = stats.pearsonr(df["highway-mpg"], df["price"])
print(pearson_coef)
print(p_value)

#Show pearson coefficient and p-value for variables peak rpm and price
df["peak-rpm"] = df["peak-rpm"].astype("float")
df.dropna(subset=["peak-rpm"], axis=0, inplace=True)
pearson_coef, p_value = stats.pearsonr(df["peak-rpm"], df["price"])
print(pearson_coef)
print(p_value)

#Show correlation between variables
corr = df.corr()
print(corr)

#Show category sizes for car body style and price using boxplot
x = df["body-style"]
y = df["price"]
sns.boxplot(x="body-style", y="price", data=df)
plt.ylim(0,)
plt.title("Category sizes for car body types and price")
plt.show()

#Show category sizes for car engine location and price using boxplot
x = df["engine-location"]
y = df["price"]
sns.boxplot(x="engine-location", y="price", data=df)
plt.ylim(0,)
plt.title("Category sizes for car engine location and price")
plt.show()

#Show category sizes for car drive wheel and price using boxplot
x = df["drive-wheels"]
y = df["price"]
sns.boxplot(x="drive-wheels", y="price", data=df)
plt.ylim(0,)
plt.title("Category sizes for car drive wheel and price")
plt.show()

#Describe the data and include object description
print(df.describe(include=["object"]))

#Predict the dependent variable price value based on highway-mpg using simple linear regression
lm = LinearRegression()
X = df[["highway-mpg"]]
Y = df["price"]

lm.fit(X, Y)
Yhat = lm.predict(X)
print(Yhat)
print(lm.intercept_)
print(lm.coef_)

print("The predicted sale value of a car with highway MPG of 30 is "+str(Yhat[30]))

#Show the R2 value for the prediction model fit for highway-mpg and price
R2 = lm.score(X, Y)
print("The R2 value for the prediction model fit for highway-mpg and price is "+str(R2))

#Compare the fit of the model for actual vs predicted values for highway-mpg and price
ax1 = sns.distplot(Y, hist=False, color="r", label="Actual Value" )
sns.distplot(Yhat, hist=False, color="b", label="Fitted Values", ax=ax1)
plt.ylim(0,)
plt.title("Compare the fit of the linear model for actual vs predited values for highway-mpg and price")
plt.show()

#Compare the fit of the model using Mean Squared Error (MSE)
mse = mean_squared_error(df["price"], Yhat)
print("The mean square error (MSE) is "+str(mse))

#Predict the dependent variable price valuse based on horsepower, curb-weight, engine-size, highway-mpg using multiple linear regression
Z = df[["horsepower", "curb-weight", "engine-size", "highway-mpg"]]
lm.fit(Z, df["price"])
Yhat = lm.predict(Z)
print(Yhat)
print(lm.intercept_)
print(lm.coef_)

#Use a pipeline to speed up the process
Input = [("scale", StandardScaler()), ("polynomial", PolynomialFeatures(include_bias=False)), ("model", LinearRegression())]
pipe = Pipeline(Input)

pipe.fit(Z, y)
ypipe = pipe.predict(Z)
print(ypipe[0:4])