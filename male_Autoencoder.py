import pandas as pd
import numpy as np
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

# Filter for male data only and reorder columns as needed
male_data = dataset[dataset['Gender'] == 'Male'].copy()
male_data = male_data[['Bust/Chest', 'Height', 'Weight', 'Waist', 'Hips', 'Body Shape Index']]

# Updated function to preprocess the male data
def preprocess_male_data(data):
    # Separate numerical and categorical features
    numerical_features = ['Bust/Chest', 'Height', 'Weight', 'Waist', 'Hips']
    categorical_feature = 'Body Shape Index'

    # Normalize numerical features using StandardScaler (Z-score normalization)
    numerical_data = data[numerical_features]
    scaler = StandardScaler()
    numerical_data_scaled = scaler.fit_transform(numerical_data)

    # One-hot encode categorical feature (Body Shape Index)
    onehot_encoder = OneHotEncoder(sparse=False)
    categorical_data_encoded = onehot_encoder.fit_transform(data[[categorical_feature]])

    # Combine all processed features in the specified order
    X_scaled = np.hstack((numerical_data_scaled, categorical_data_encoded))
    
    return X_scaled, scaler, onehot_encoder

# Preprocess the male data
X_scaled_male, scaler_male, onehot_encoder_male = preprocess_male_data(male_data)

def train_and_generate_size_chart_male(X_scaled, scaler, num_numerical, num_categorical, latent_dim=20, n_clusters=4, epochs=1000, batch_size=64, learning_rate=0.001):
    # Train autoencoder and perform hierarchical clustering
    decoded_centroids_original, clusters, latent_representations_scaled = train_autoencoder_and_cluster_male(
        X_scaled, scaler, num_numerical, num_categorical, latent_dim, n_clusters, epochs, batch_size, learning_rate
    )

    # Generate size chart sorted by chest size or any relevant size feature
    size_chart = pd.DataFrame(decoded_centroids_original, columns=['Bust/Chest', 'Height', 'Weight', 'Waist', 'Hips', 'Body Shape Index'])

    # Sort size chart based on 'Bust/Chest' (or another relevant feature if needed)
    size_chart = size_chart.sort_values(by='Bust/Chest', ascending=True)
    
    # Assign size labels ['S', 'M', 'L', 'XL'] based on sorted order
    size_labels = ['S', 'M', 'L', 'XL']
    size_chart['Size'] = size_labels

    # Print the sorted size chart
    print("Size Chart (Sorted):")
    print(size_chart)

    return size_chart, latent_representations_scaled


def deprocess_male(decoded_centroids, scaler, num_numerical, onehot_encoder):
    """
    Deprocesses the decoded centroids to return the data back to its original scale and format.

    Parameters:
    - decoded_centroids: The decoded data from the autoencoder in its current form.
    - scaler: The StandardScaler object used for initial scaling of the numerical data.
    - num_numerical: The number of numerical features that were scaled.
    - onehot_encoder: The OneHotEncoder object used for encoding the categorical data.

    Returns:
    - decoded_centroids_original: The centroids transformed back to the original scale and format.
    """

    # Separate numerical and categorical parts of the decoded centroids
    decoded_numerical = decoded_centroids[:, :num_numerical]
    decoded_categorical = decoded_centroids[:, num_numerical:]

    # Inverse transform the numerical data back to its original scale
    decoded_numerical_original = scaler.inverse_transform(decoded_numerical)

    # Decode the one-hot encoded categorical data back to the original categorical labels
    decoded_categorical_original = onehot_encoder.inverse_transform(decoded_categorical)

    # Concatenate numerical and categorical data back together
    decoded_centroids_original = np.hstack((decoded_numerical_original, decoded_categorical_original))

    return decoded_centroids_original

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import AgglomerativeClustering
import tensorflow as tf
from tensorflow.keras import layers, models

def train_autoencoder_and_cluster_male(X_scaled, scaler, num_numerical, num_categorical, latent_dim=20, n_clusters=4, epochs=1000, batch_size=64, learning_rate=0.001):
    input_dim = X_scaled.shape[1]

    # Updated Autoencoder Model Architecture for males
    input_layer = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(512, activation='relu')(input_layer)
    encoded = layers.Dense(256, activation='relu')(encoded)
    encoded = layers.Dense(128, activation='relu')(encoded)
    encoded = layers.Dense(64, activation='relu')(encoded)
    encoded = layers.Dense(32, activation='relu')(encoded)
    encoded = layers.Dense(16, activation='relu')(encoded)
    latent = layers.Dense(latent_dim, activation='relu')(encoded)
    
    # Decoder layers for both numerical and categorical outputs
    decoded_common = layers.Dense(16, activation='relu')(latent)
    decoded_common = layers.Dense(32, activation='relu')(decoded_common)
    decoded_common = layers.Dense(64, activation='relu')(decoded_common)
    decoded_common = layers.Dense(128, activation='relu')(decoded_common)
    decoded_common = layers.Dense(256, activation='relu')(decoded_common)
    decoded_common = layers.Dense(512, activation='relu')(decoded_common)

    # Output layer for numerical features (using linear activation)
    decoded_numerical = layers.Dense(num_numerical, activation='linear')(decoded_common)

    # Output layer for categorical features (using softmax activation)
    decoded_categorical = layers.Dense(num_categorical, activation='softmax')(decoded_common)

    # Concatenate numerical and categorical outputs
    decoded_output = layers.Concatenate()([decoded_numerical, decoded_categorical])

    autoencoder = models.Model(input_layer, decoded_output)
    encoder = models.Model(input_layer, latent)

    # Reuse decoder layers from autoencoder to construct the decoder
    decoder_input = layers.Input(shape=(latent_dim,))
    
    # Start with the decoder input and use the same layers from autoencoder
    decoded_layer = decoder_input
    decoded_layer = autoencoder.layers[-9](decoded_layer)
    decoded_layer = autoencoder.layers[-8](decoded_layer)
    decoded_layer = autoencoder.layers[-7](decoded_layer)
    decoded_layer = autoencoder.layers[-6](decoded_layer)
    decoded_layer = autoencoder.layers[-5](decoded_layer)
    decoded_layer = autoencoder.layers[-4](decoded_layer)

    # Output layers for numerical and categorical data
    decoded_output_numerical = autoencoder.layers[-3](decoded_layer)
    decoded_output_categorical = autoencoder.layers[-2](decoded_layer)

    # Concatenate outputs to form the final output layer
    decoded_output = layers.Concatenate()([decoded_output_numerical, decoded_output_categorical])

    # Create the decoder model
    decoder = models.Model(decoder_input, decoded_output)

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
    decoded_centroids_original = deprocess_male(decoded_centroids, scaler, num_numerical, onehot_encoder_male)

    return decoded_centroids_original, clusters, latent_representations_scaled

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# Function to train autoencoder, perform hierarchical clustering, and generate the size chart
def train_and_generate_size_chart_male(X_scaled, scaler, num_numerical, num_categorical, latent_dim=20, n_clusters=4, epochs=1000, batch_size=64, learning_rate=0.001):
    # Train autoencoder and perform hierarchical clustering
    decoded_centroids_original, clusters, latent_representations_scaled = train_autoencoder_and_cluster_male(
        X_scaled, scaler, num_numerical, num_categorical, latent_dim, n_clusters, epochs, batch_size, learning_rate
    )

    # Generate size chart sorted by chest size or any relevant size feature
    size_chart = pd.DataFrame(decoded_centroids_original, columns=['Bust/Chest', 'Height', 'Weight', 'Waist', 'Hips', 'Body Shape Index'])

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

# Train model and generate size chart for male data
num_numerical = 5  # Number of numerical features for male data (excluding Cup Size)
num_categorical = onehot_encoder_male.categories_[0].shape[0]  # Number of categories for 'Body Shape Index'
X_scaled_male, scaler_male, onehot_encoder_male = preprocess_male_data(male_data)
size_chart_male, latent_representations_scaled_male = train_and_generate_size_chart_male(
    X_scaled_male, scaler_male, num_numerical, num_categorical, latent_dim=20, n_clusters=4, epochs=1000, batch_size=32, learning_rate=0.001
)

# Plot dendrogram for male data
plot_dendrogram(latent_representations_scaled_male)

