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

# Focus on the Waist measurement for clustering
male_waist = male_data[['Waist']].copy()
female_waist = female_data[['Waist']].copy()

# Standardize the waist data
scaler_male_waist = StandardScaler()
male_waist_scaled = scaler_male_waist.fit_transform(male_waist)

scaler_female_waist = StandardScaler()
female_waist_scaled = scaler_female_waist.fit_transform(female_waist)

# Perform K-Means clustering on waist measurements only
kmeans_male_waist = KMeans(n_clusters=6, random_state=42)
male_waist_clusters = kmeans_male_waist.fit_predict(male_waist_scaled)
male_waist_centroids = kmeans_male_waist.cluster_centers_

kmeans_female_waist = KMeans(n_clusters=6, random_state=42)
female_waist_clusters = kmeans_female_waist.fit_predict(female_waist_scaled)
female_waist_centroids = kmeans_female_waist.cluster_centers_

# Convert waist centroids back to the original scale
male_waist_centroids_original_scale = scaler_male_waist.inverse_transform(male_waist_centroids)
female_waist_centroids_original_scale = scaler_female_waist.inverse_transform(female_waist_centroids)

# Output the waist centroids
print("Male Waist Centroids (Original Scale):")
print(pd.DataFrame(male_waist_centroids_original_scale, columns=['Waist']))

print("\nFemale Waist Centroids (Original Scale):")
print(pd.DataFrame(female_waist_centroids_original_scale, columns=['Waist']))

