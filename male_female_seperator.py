import pandas as pd

# Load the original dataset
dataset = pd.read_csv('body_measurements_dataset.csv')

# Separate the data by gender
male_data = dataset[dataset['Gender'] == 'Male'].copy()
female_data = dataset[dataset['Gender'] == 'Female'].copy()

# Save the separated datasets to CSV files
male_data.to_csv('male_data.csv', index=False)
female_data.to_csv('female_data.csv', index=False)

# Print confirmation
print(f"Male dataset saved as 'male_data.csv' with {male_data.shape[0]} rows.")
print(f"Female dataset saved as 'female_data.csv' with {female_data.shape[0]} rows.")
