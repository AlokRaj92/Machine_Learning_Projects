import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv(r"D:\\Deployment\\USA_housing_Price\\USA_Housing.csv")
df.head()

#df.drop(['state'], axis = 1, inplace = True)
# LabelEncoder
le = LabelEncoder()
df['state'] = le.fit_transform(df['state'])

# Separate target variable (Price) and feature variables
y = df['Price']  # Target variable
x = df.drop(['Price'], axis=1)

# Apply MinMaxScaler only to numeric features (excluding 'state')
scaler = MinMaxScaler()
numeric_columns = x.drop('state', axis=1).columns  # Exclude 'state' column for scaling
x_scaled_numeric = scaler.fit_transform(x[numeric_columns])

# Create a DataFrame with scaled numeric columns and add back the 'state' column
x_scaled = pd.DataFrame(x_scaled_numeric, columns=numeric_columns)
x_scaled['state'] = x['state']  # Add the encoded 'state' column back

# Splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.30, random_state=42)


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)

import pickle
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

import os
os.getcwd()