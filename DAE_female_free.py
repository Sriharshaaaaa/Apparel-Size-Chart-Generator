import pandas as pd
import numpy as np
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

# Apply the height conversion
dataset['Height'] = dataset['Height'].apply(convert_height_to_inches)

# Filter for female data only and reorder columns to include 'Cup Size' as an ordinal variable
female_data = dataset[dataset['Gender'] == 'Female'].copy()
female_data = female_data[['Bust/Chest', 'Height', 'Weight', 'Waist', 'Hips', 'Cup Size']]

# Function to preprocess the female data (including 'Cup Size' as an ordinal variable)
def preprocess_female_data(data):
    # Separate numerical and ordinal features
    numerical_features = ['Bust/Chest', 'Height', 'Weight', 'Waist', 'Hips']
    ordinal_feature = 'Cup Size'

    # Normalize numerical features using StandardScaler (Z-score normalization)
    numerical_data = data[numerical_features]
    scaler = StandardScaler()
    numerical_data_scaled = scaler.fit_transform(numerical_data)

    # Encode ordinal feature (Cup Size) using predefined mapping
    cup_size_mapping = {'AA': -3, 'A': -2, 'B': -1, 'C': 0, 'D': 1, 'DD': 2, 'E': 3, 'F': 4}
    ordinal_data = data[ordinal_feature].map(cup_size_mapping).values.reshape(-1, 1)

    # Combine numerical and ordinal data
    X_scaled = np.hstack((numerical_data_scaled, ordinal_data))
    
    return X_scaled, scaler

# Preprocess the female data
X_scaled_female, scaler_female = preprocess_female_data(female_data)

def deprocess_female(decoded_centroids, scaler, num_numerical):
    """
    Deprocesses the decoded centroids to return the data back to its original scale.

    Parameters:
    - decoded_centroids: The decoded data from the autoencoder in its current form.
    - scaler: The StandardScaler object used for initial scaling of the numerical data.
    - num_numerical: The number of numerical features that were scaled.

    Returns:
    - decoded_centroids_original: The centroids transformed back to the original scale and format.
    """
    
    # Ensure decoded_centroids is a NumPy array
    decoded_centroids = np.array(decoded_centroids)

    # Separate numerical and ordinal parts of the decoded centroids
    decoded_numerical = decoded_centroids[:, :num_numerical]  # Numerical data
    decoded_ordinal = decoded_centroids[:, num_numerical:]  # Ordinal data (Cup Size)

    # Inverse transform the numerical data back to its original scale
    decoded_numerical_original = scaler.inverse_transform(decoded_numerical)

    # Concatenate numerical and ordinal data back together
    decoded_centroids_original = np.hstack((decoded_numerical_original, decoded_ordinal))

    return decoded_centroids_original

import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.cluster import AgglomerativeClustering

def train_autoencoder_and_cluster_female(X_scaled, scaler, num_numerical, latent_dim=20, n_clusters=4, epochs=1000, batch_size=64, learning_rate=0.001):
    input_dim = X_scaled.shape[1]

    # Autoencoder Model Architecture
    input_layer = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(512, activation='relu')(input_layer)
    encoded = layers.Dense(256, activation='relu')(encoded)
    encoded = layers.Dense(128, activation='relu')(encoded)
    encoded = layers.Dense(64, activation='relu')(encoded)
    encoded = layers.Dense(32, activation='relu')(encoded)
    encoded = layers.Dense(16, activation='relu')(encoded)
    latent = layers.Dense(latent_dim, activation='relu')(encoded)
    
    # Decoder layers for numerical and ordinal outputs
    decoded_common = layers.Dense(16, activation='relu')(latent)
    decoded_common = layers.Dense(32, activation='relu')(decoded_common)
    decoded_common = layers.Dense(64, activation='relu')(decoded_common)
    decoded_common = layers.Dense(128, activation='relu')(decoded_common)
    decoded_common = layers.Dense(256, activation='relu')(decoded_common)
    decoded_common = layers.Dense(512, activation='relu')(decoded_common)

    # Output layer for all features (using linear activation for numerical and ordinal data)
    decoded_output = layers.Dense(num_numerical + 1, activation='linear')(decoded_common)

    autoencoder = models.Model(input_layer, decoded_output)
    encoder = models.Model(input_layer, latent)

    # Reuse decoder layers from autoencoder to construct the decoder
    decoder_input = layers.Input(shape=(latent_dim,))
    decoded_layer = decoder_input
    for i in range(-7, 0):  # Reuse layers from the autoencoder
        decoded_layer = autoencoder.layers[i](decoded_layer)

    # Create the decoder model
    decoder = models.Model(decoder_input, decoded_layer)

    # Compile the model with a custom learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    autoencoder.compile(optimizer=optimizer, loss='mse')

    # Train the autoencoder
    autoencoder.fit(X_scaled, X_scaled, epochs=epochs, batch_size=batch_size, shuffle=True)

    # Extract the latent representations
    latent_representations = encoder.predict(X_scaled)

    # Normalize the latent representations
    latent_scaler = StandardScaler()
    latent_representations_scaled = latent_scaler.fit_transform(latent_representations)

    # Hierarchical Clustering in Latent Space
    hierarchical_clustering = AgglomerativeClustering(n_clusters=n_clusters)
    clusters = hierarchical_clustering.fit_predict(latent_representations_scaled)

    # Calculate the cluster centroids manually
    cluster_centroids = np.array([latent_representations_scaled[clusters == i].mean(axis=0) for i in range(n_clusters)])

    # Denormalize the cluster centroids to the original latent space
    cluster_centroids_denormalized = latent_scaler.inverse_transform(cluster_centroids)

    # Decode the cluster centroids to get the original space values
    decoded_centroids = decoder.predict(cluster_centroids_denormalized)
    decoded_centroids_original = deprocess_female(decoded_centroids, scaler, num_numerical)

    return decoded_centroids_original, clusters, latent_representations_scaled

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# Function to train autoencoder, perform hierarchical clustering, and generate the size chart
def train_and_generate_size_chart_female(X_scaled, scaler, num_numerical, latent_dim=20, n_clusters=4, epochs=1000, batch_size=64, learning_rate=0.001):
    # Train autoencoder and perform hierarchical clustering
    decoded_centroids_original, clusters, latent_representations_scaled = train_autoencoder_and_cluster_female(
        X_scaled, scaler, num_numerical, latent_dim, n_clusters, epochs, batch_size, learning_rate
    )

    # Generate size chart sorted by 'Bust/Chest' or any other relevant feature
    size_chart = pd.DataFrame(decoded_centroids_original, columns=['Bust/Chest', 'Height', 'Weight', 'Waist', 'Hips', 'Cup Size'])

    # Sort size chart based on 'Bust/Chest' (or another relevant feature if needed)
    size_chart = size_chart.sort_values(by='Bust/Chest', ascending=True)
    
    # Assign size labels ['S', 'M', 'L', 'XL'] based on sorted order
    size_labels = ['S', 'M', 'L', 'XL']
    size_chart['Size'] = size_labels

    # Print the sorted size chart
    print("Size Chart (Sorted):")
    print(size_chart)

    return size_chart, latent_representations_scaled

# Plot dendrogram for hierarchical clustering
def plot_dendrogram(latent_representations_scaled):
    plt.figure(figsize=(10, 7))
    Z = linkage(latent_representations_scaled, 'ward')
    dendrogram(Z)
    plt.title('Dendrogram for Hierarchical Clustering')
    plt.xlabel('Samples')
    plt.ylabel('Distance')
    plt.show()

# Train model and generate size chart for female data
X_scaled_female, scaler_female = preprocess_female_data(female_data)
size_chart_female, latent_representations_scaled_female = train_and_generate_size_chart_female(
    X_scaled_female, scaler_female, num_numerical=5, latent_dim=20, n_clusters=
    4, epochs=100, batch_size=64, learning_rate=0.001
)

# Plot dendrogram
plot_dendrogram(latent_representations_scaled_female)
