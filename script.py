import pandas as pd

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
path = "C://Projects/datascience/new_dataset.csv"

#Export new dataset to new file
df.to_csv(path)

#Show datatypes
print(df.dtypes)

#Show description of the data
print(df.describe(include="all"))

#Show summary of the data
print(df.info())

#Drop na values
df.dropna(subset=["price"], axis=0)
print(df)