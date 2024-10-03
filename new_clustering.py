import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np

# Load the dataset
file_path = 'body_measurements_dataset.csv'
df = pd.read_csv(file_path)

# Function to convert height from feet and inches to inches
def convert_height(height_str):
    if pd.isna(height_str):
        return np.nan
    try:
        feet, inches = height_str.split("'")
        inches = inches.replace('"', '')
        return int(feet) * 12 + int(inches)
    except ValueError:
        return np.nan

# Apply the height conversion
df['Height'] = df['Height'].apply(convert_height)

# Map cup sizes to numeric values for females only
cup_size_mapping = {
    'AA': 1,
    'A': 2,
    'B': 3,
    'C': 4,
    'DD': 5,
    'D': 6,
    'E': 7,
    'F': 8
}

# Only map cup sizes for females
df.loc[df['Gender'] == 'Female', 'Cup Size'] = df.loc[df['Gender'] == 'Female', 'Cup Size'].map(cup_size_mapping)

# Handle missing values: Fill only numeric columns with their mean
numeric_columns = df.select_dtypes(include=[np.number]).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

# Separate the dataset by gender
df_male = df[df['Gender'] == 'Male'].copy()
df_female = df[df['Gender'] == 'Female'].copy()

# Select relevant features for different apparel categories
def get_features_for_category(df, category):
    if category == 'T-shirt':
        return df[['Bust/Chest']]
    elif category == 'Skirt':
        return df[['Waist', 'Hips']]
    elif category == 'Pants':
        return df[['Waist', 'Hips']]
    elif category == 'Bra':
        return df[['Bust/Chest', 'Cup Size']]
    else:
        return df[['Bust/Chest', 'Waist', 'Hips']]

# Normalize the data and store the scaling parameters
def normalize_data(df):
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(df)
    return normalized_data, scaler.mean_, scaler.scale_

# De-normalize the centroid values
def denormalize_centroids(centroids, mean, scale):
    return centroids * scale + mean

# Apply K-means clustering
def apply_clustering(data, num_clusters=4):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(data)
    return kmeans

# Assign size labels based on clusters
def assign_size_labels(df, category, cluster_centers):
    # Calculate the average size for each cluster
    avg_sizes = {i: np.mean(cluster_centers[i]) for i in range(len(cluster_centers))}
    
    # Sort clusters by average size
    sorted_clusters = sorted(avg_sizes, key=avg_sizes.get)
    
    # Map sorted clusters to sizes
    size_mapping = {sorted_clusters[i]: size for i, size in enumerate(['S', 'M', 'L', 'XL'])}
    
    # Apply size mapping to the dataframe
    df[f'{category}_size_label'] = df[f'{category}_size'].map(size_mapping)
    return df, size_mapping

# Generate size chart based on clustering and de-normalize centroids
def generate_size_chart(df, category, num_clusters=4):
    features = get_features_for_category(df, category)
    normalized_data, mean, scale = normalize_data(features)
    kmeans = apply_clustering(normalized_data, num_clusters)
    
    df.loc[:, f'{category}_size'] = kmeans.labels_
    
    # Assign size labels
    df, size_mapping = assign_size_labels(df, category, kmeans.cluster_centers_)
    
    # De-normalize the centroids
    centroids = denormalize_centroids(kmeans.cluster_centers_, mean, scale)
    
    # Create a dictionary to store de-normalized centroid values with size labels
    centroids = {size_mapping[i]: centroids[i] for i in range(num_clusters)}
    
    return df, centroids

# Categories to process for both male and female
categories = ['T-shirt', 'Skirt', 'Pants', 'Bra']

# Process each category for males
for category in categories:
    if category == 'Bra':  # Skip Bra for males
        continue
    df_male, centroids_male = generate_size_chart(df_male, category)
    print(f"{category} Size Chart (Male):")
    for size, centroid in sorted(centroids_male.items()):
        print(f"Size: {size}, Centroid: {centroid}")

# Process each category for females
for category in categories:
    df_female, centroids_female = generate_size_chart(df_female, category)
    print(f"\n{category} Size Chart (Female):")
    for size, centroid in sorted(centroids_female.items()):
        print(f"Size: {size}, Centroid: {centroid}")

# Save the updated dataset with size labels for all categories
df_male.to_csv('updated_male_dataset.csv', index=False)
df_female.to_csv('updated_female_dataset.csv', index=False)
