import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load the dataset
data = pd.read_csv("C:/Users/prans/Downloads/Walmart_sales (1).csv")

# Feature selection (excluding Date and the target Weekly_Sales)
X = data[['Store', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']]
y = data['Weekly_Sales']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model to a pickle file
with open('walmart_sales_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Demonstrate loading the model from the pickle file
with open('walmart_sales_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Make predictions using the loaded model
predictions = loaded_model.predict(X_test[:5])

# Output predictions
print('Predictions for the first 5 test samples:', predictions)

# If you want to save this script, you can manually save it as a .py file
