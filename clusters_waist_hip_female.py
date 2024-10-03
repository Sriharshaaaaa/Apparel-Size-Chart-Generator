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

# Focus on female data
female_data = dataset[dataset['Gender'] == 'Female'].copy()
cup_size_mapping = {'AA': 1, 'A': 2, 'B': 3, 'C': 4, 'D': 5, 'DD': 6, 'E': 7, 'F': 8}
female_data['Cup Size'] = female_data['Cup Size'].map(cup_size_mapping)
# Focus on Waist and Hip measurements for clustering
female_waist_hip = female_data[['Bust/Chest', 'Cup Size']].copy()

# Standardize the waist and hip data
scaler_female_waist_hip = StandardScaler()
female_waist_hip_scaled = scaler_female_waist_hip.fit_transform(female_waist_hip)

# Perform K-Means clustering on waist and hip measurements
kmeans_female_waist_hip = KMeans(n_clusters=6, random_state=42)
female_waist_hip_clusters = kmeans_female_waist_hip.fit_predict(female_waist_hip_scaled)
female_waist_hip_centroids = kmeans_female_waist_hip.cluster_centers_

# Convert centroids back to the original scale
female_waist_hip_centroids_original_scale = scaler_female_waist_hip.inverse_transform(female_waist_hip_centroids)

# Output the centroids
print("Female Waist and Hip Centroids (Original Scale):")
print(pd.DataFrame(female_waist_hip_centroids_original_scale, columns=['Waist', 'Hips']))

# Plotting the clusters
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.scatter(female_waist_hip_scaled[:, 0], female_waist_hip_scaled[:, 1], c=female_waist_hip_clusters, cmap='viridis')
plt.scatter(female_waist_hip_centroids[:, 0], female_waist_hip_centroids[:, 1], s=300, c='red', marker='X')  # Centroids
plt.title('Female Waist and Hip Clusters')
plt.xlabel('Standardized Waist')
plt.ylabel('Standardized Hips')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()
