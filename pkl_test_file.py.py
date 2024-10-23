import pandas as pd
import pickle

# 1. Define the new sample dataset (replace this with your actual new data)
new_data = {
    'Store': [1, 2, 3],                 # Store number (same feature as used in training)
    'Holiday_Flag': [0, 1, 0],          # Holiday flag (binary feature)
    'Temperature': [85.0, 80.0, 78.0],  # Continuous feature (Temperature in Fahrenheit)
    'Fuel_Price': [2.57, 2.67, 2.77],   # Continuous feature (Fuel price)
    'CPI': [211.0, 215.5, 210.7],       # Continuous feature (Consumer Price Index)
    'Unemployment': [6.5, 6.7, 6.9]     # Continuous feature (Unemployment rate)
}

# 2. Convert the new data to a DataFrame
new_data_df = pd.DataFrame(new_data)

# 3. Load the saved model from the pickle file
with open('walmart_sales_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# 4. Preprocess the new data if necessary
# Since this is a simple regression model, and we don't have scaling in the original setup, 
# we directly use the new data without needing to transform the features

# 5. Make predictions using the loaded model
predictions = loaded_model.predict(new_data_df)

# Output the predictions
print('Predictions for the new data:', predictions)

# 6. Optionally, you can save the predictions to a CSV file along with the original new data
new_data_df['Predicted_Weekly_Sales'] = predictions
new_data_df.to_csv('new_walmart_sales_with_predictions.csv', index=False)
print('Predictions saved to new_walmart_sales_with_predictions.csv')
