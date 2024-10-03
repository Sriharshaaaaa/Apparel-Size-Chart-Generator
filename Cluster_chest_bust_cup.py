import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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

# Apply the height conversion (optional, if not already done)
dataset['Height'] = dataset['Height'].apply(convert_height_to_inches)

# Separate the data by gender
male_data = dataset[dataset['Gender'] == 'Male'].copy()
female_data = dataset[dataset['Gender'] == 'Female'].copy()

# Focus on the Bust/Chest measurement for clustering
male_chest = male_data[['Bust/Chest']].copy()
female_chest = female_data[['Bust/Chest', 'Cup Size']].copy()

# Numerical mapping for Cup Size in the female dataset
cup_size_mapping = {'A': 1, 'AA': 2, 'B': 3, 'C': 4, 'D': 5, 'DD': 6, 'E': 7, 'F': 8}
female_chest['Cup Size'] = female_chest['Cup Size'].map(cup_size_mapping)

# Standardize the chest data
scaler_male_chest = StandardScaler()
male_chest_scaled = scaler_male_chest.fit_transform(male_chest)

scaler_female_chest = StandardScaler()
female_chest_scaled = scaler_female_chest.fit_transform(female_chest)

# Perform K-Means clustering on chest measurements only
kmeans_male_chest = KMeans(n_clusters=6, random_state=42)
male_chest_clusters = kmeans_male_chest.fit_predict(male_chest_scaled)
male_chest_centroids = kmeans_male_chest.cluster_centers_

kmeans_female_chest = KMeans(n_clusters=6, random_state=42)
female_chest_clusters = kmeans_female_chest.fit_predict(female_chest_scaled)
female_chest_centroids = kmeans_female_chest.cluster_centers_

# Convert chest centroids back to the original scale
male_chest_centroids_original_scale = scaler_male_chest.inverse_transform(male_chest_centroids)
female_chest_centroids_original_scale = scaler_female_chest.inverse_transform(female_chest_centroids)

# Output the chest centroids
print("Male Chest Centroids (Original Scale):")
print(pd.DataFrame(male_chest_centroids_original_scale, columns=['Bust/Chest']))

print("\nFemale Chest Centroids (Original Scale with Cup Size):")
print(pd.DataFrame(female_chest_centroids_original_scale, columns=['Bust/Chest', 'Cup Size']))
