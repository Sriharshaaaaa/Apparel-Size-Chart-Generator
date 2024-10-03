import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load the dataset
dataset = pd.read_csv('body_measurements_dataset.csv')

# Function to convert height from feet'inches" to inches (if needed)
def convert_height_to_inches(height):
    if isinstance(height, str):
        parts = height.split("'")
        feet = int(parts[0])
        inches = int(parts[1].replace('"', ''))
        return feet * 12 + inches
    return np.nan

# Apply the height conversion
dataset['Height'] = dataset['Height'].apply(convert_height_to_inches)

# Separate the data by gender
male_data = dataset[dataset['Gender'] == 'Male'].copy()
female_data = dataset[dataset['Gender'] == 'Female'].copy()

# Define features
numerical_features = ['Bust/Chest', 'Height', 'Weight', 'Waist', 'Hips']
categorical_features = ['Body Shape Index', 'Cup Size']

# Function to preprocess data (one-hot encoding and scaling)
def preprocess_data(df, numerical_features, categorical_features):
    # One-hot encode categorical features
    onehot_encoder = OneHotEncoder(sparse=False)
    encoded_categorical = onehot_encoder.fit_transform(df[categorical_features])
    
    # Combine numerical features with encoded categorical features
    numerical_data = df[numerical_features].values
    X = np.hstack([numerical_data, encoded_categorical])
    
    # Return the ordered columns for checking
    column_order = numerical_features + list(onehot_encoder.get_feature_names_out(categorical_features))
    return X, column_order

# Check column order for male data
_, male_column_order = preprocess_data(male_data, numerical_features, categorical_features)

# Check column order for female data
_, female_column_order = preprocess_data(female_data, numerical_features, categorical_features)

print("Male Data Column Order:")
print(male_column_order)

print("\nFemale Data Column Order:")
print(female_column_order)
