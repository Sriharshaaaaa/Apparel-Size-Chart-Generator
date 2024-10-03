import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
dataset = pd.read_csv('body_measurements_dataset.csv')

# Function to convert height from feet'inches" to inches
def convert_height_to_inches(height):
    if isinstance(height, str):
        parts = height.split("'")
        feet = int(parts[0])
        inches = int(parts[1].replace('"', ''))
        return feet * 12 + inches
    return np.nan

# Apply the height conversion
dataset['Height'] = dataset['Height'].apply(convert_height_to_inches)

# Ensure that the data is correctly separated by gender
male_data = dataset[dataset['Gender'] == 'Male'].copy()
female_data = dataset[dataset['Gender'] == 'Female'].copy()

# Numerical mapping for Cup Size in the female dataset
cup_size_mapping = {'A': 1, 'AA': 2, 'B': 3, 'C': 4, 'D': 5, 'DD': 6, 'E': 7, 'F': 8}
female_data['Cup Size'] = female_data['Cup Size'].map(cup_size_mapping)

# Drop the 'Gender' column for both datasets, and 'Cup Size' for males
male_data = male_data.drop(columns=['Gender', 'Cup Size'])
female_data = female_data.drop(columns=['Gender'])

# Double-check for any NaN values after mapping (though none should be present)
print("Checking for NaN values in male dataset:\n", male_data.isna().sum())
print("Checking for NaN values in female dataset:\n", female_data.isna().sum())

# Standardize the data
scaler_male = StandardScaler()
male_data_scaled = scaler_male.fit_transform(male_data)

scaler_female = StandardScaler()
female_data_scaled = scaler_female.fit_transform(female_data)

# Perform K-Means clustering
kmeans_male = KMeans(n_clusters=4, random_state=42)
male_clusters = kmeans_male.fit_predict(male_data_scaled)
male_centroids = kmeans_male.cluster_centers_

kmeans_female = KMeans(n_clusters=4, random_state=42)
female_clusters = kmeans_female.fit_predict(female_data_scaled)
female_centroids = kmeans_female.cluster_centers_

# Convert centroids back to the original scale
male_centroids_original_scale = scaler_male.inverse_transform(male_centroids)
female_centroids_original_scale = scaler_female.inverse_transform(female_centroids)

# Output the centroids
print("Male Centroids (Original Scale):")
print(pd.DataFrame(male_centroids_original_scale, columns=male_data.columns))

print("\nFemale Centroids (Original Scale with Cup Size):")
print(pd.DataFrame(female_centroids_original_scale, columns=female_data.columns))
