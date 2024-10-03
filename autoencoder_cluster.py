import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

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

# Function to preprocess data (one-hot encoding and scaling)
def preprocess_data(df, numerical_features, categorical_features):
    # One-hot encode categorical features
    onehot_encoder = OneHotEncoder(sparse=False)
    encoded_categorical = onehot_encoder.fit_transform(df[categorical_features])
    
    # Combine numerical features with encoded categorical features
    numerical_data = df[numerical_features].values
    X = np.hstack([numerical_data, encoded_categorical])
    
    # Normalize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, scaler, onehot_encoder

# Function to split data into train, validation, and test sets
def split_data(X, test_size=0.2, validation_size=0.1):
    X_train, X_temp = train_test_split(X, test_size=(test_size + validation_size), random_state=42)
    X_val, X_test = train_test_split(X_temp, test_size=(test_size / (test_size + validation_size)), random_state=42)
    return X_train, X_val, X_test

def train_autoencoder_and_cluster(X_scaled, scaler, num_numerical, num_categorical, latent_dim=20, n_clusters=4, epochs=1000, batch_size=64, learning_rate=0.001):
    input_dim = X_scaled.shape[1]

    # Autoencoder Model with increased complexity
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

    # Output layer for numerical features
    decoded_numerical = layers.Dense(num_numerical, activation='leaky_relu')(decoded_common)

    # Output layer for categorical features
    decoded_categorical = layers.Dense(num_categorical, activation='sigmoid')(decoded_common)

    # Concatenate numerical and categorical outputs
    decoded_output = layers.Concatenate()([decoded_numerical, decoded_categorical])

    autoencoder = models.Model(input_layer, decoded_output)
    encoder = models.Model(input_layer, latent)

    # Reuse decoder layers from autoencoder to construct the decoder
    decoder_input = layers.Input(shape=(latent_dim,))
    
    # Start with the decoder input and use the same layers from autoencoder
    decoded_layer = decoder_input
    decoded_layer = autoencoder.layers[-9](decoded_layer)  # Corresponds to the first decoder layer
    decoded_layer = autoencoder.layers[-8](decoded_layer)  # Corresponds to the second decoder layer
    decoded_layer = autoencoder.layers[-7](decoded_layer)  # Corresponds to the third decoder layer
    decoded_layer = autoencoder.layers[-6](decoded_layer)  # Corresponds to the fourth decoder layer
    decoded_layer = autoencoder.layers[-5](decoded_layer)  # Corresponds to the fifth decoder layer
    decoded_layer = autoencoder.layers[-4](decoded_layer)  # Corresponds to the sixth decoder layer

    # Output layers for numerical and categorical data
    decoded_output_numerical = autoencoder.layers[-3](decoded_layer)
    decoded_output_categorical = autoencoder.layers[-2](decoded_layer)

    # Concatenate outputs to form the final output layer
    decoded_output = layers.Concatenate()([decoded_output_numerical, decoded_output_categorical])

    # Create the decoder model
    decoder = models.Model(decoder_input, decoded_output)

    print("Autoencoder Model Summary:")
    autoencoder.summary()

    print("\nDecoder Model Summary:")
    decoder.summary()
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

    # Clustering in Latent Space
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(latent_representations_scaled)

    # Get the cluster centroids in latent space
    cluster_centroids = kmeans.cluster_centers_

    # Denormalize the cluster centroids to the original latent space
    cluster_centroids_denormalized = latent_scaler.inverse_transform(cluster_centroids)

    # Decode the cluster centroids to get the original space values
    decoded_centroids = decoder.predict(cluster_centroids_denormalized)
    decoded_centroids_original = scaler.inverse_transform(decoded_centroids)

    return decoded_centroids_original, clusters, latent_representations_scaled


# Function to generate the size chart with all features
def generate_size_chart(decoded_centroids_original, feature_names):
    # Adjust feature names to match the number of columns in decoded_centroids_original
    size_chart = pd.DataFrame(decoded_centroids_original, columns=feature_names[:decoded_centroids_original.shape[1]])
    size_chart['Cluster'] = range(1, len(size_chart) + 1)
    return size_chart

# Define features
numerical_features = ['Bust/Chest', 'Height', 'Weight', 'Waist', 'Hips']
categorical_features = ['Body Shape Index', 'Cup Size']

# Male data processing
X_male_scaled, male_scaler, male_onehot_encoder = preprocess_data(male_data, numerical_features, categorical_features)
X_male_train, X_male_val, X_male_test = split_data(X_male_scaled)

# Get number of numerical and categorical features
num_numerical = len(numerical_features)
num_categorical = len(male_onehot_encoder.get_feature_names_out(categorical_features))

# Train autoencoder on the training set and cluster on the validation set
decoded_centroids_male, clusters_male, latent_representations_male = train_autoencoder_and_cluster(
    X_male_train, male_scaler, num_numerical, num_categorical, latent_dim=3, n_clusters=4, epochs=150, batch_size=64, learning_rate=0.001)

# Female data processing
X_female_scaled, female_scaler, female_onehot_encoder = preprocess_data(female_data, numerical_features, categorical_features)
X_female_train, X_female_val, X_female_test = split_data(X_female_scaled)

# Get number of numerical and categorical features
num_categorical_female = len(female_onehot_encoder.get_feature_names_out(categorical_features))

# Train autoencoder on the training set and cluster on the validation set
decoded_centroids_female, clusters_female, latent_representations_female = train_autoencoder_and_cluster(
    X_female_train, female_scaler, num_numerical, num_categorical_female, latent_dim=20, n_clusters=4, epochs=1000, batch_size=64, learning_rate=0.001)

# Generate size charts with all features
# Get feature names including the one-hot encoded features
feature_names_male = numerical_features + list(male_onehot_encoder.get_feature_names_out(categorical_features))
feature_names_female = numerical_features + list(female_onehot_encoder.get_feature_names_out(categorical_features))

male_size_chart = generate_size_chart(decoded_centroids_male, feature_names_male)
female_size_chart = generate_size_chart(decoded_centroids_female, feature_names_female)

print("Initial Male Size Chart (All Features):")
print(male_size_chart)

print("\nInitial Female Size Chart (All Features):")
print(female_size_chart)

# Optional: Visualize the clusters in latent space
def plot_clusters(latent_representations, clusters, cluster_centroids):
    plt.scatter(latent_representations[:, 0], latent_representations[:, 1], c=clusters, cmap='viridis')
    plt.scatter(cluster_centroids[:, 0], cluster_centroids[:, 1], s=300, c='red', marker='X')  # Centroids
    plt.title('Clusters in Latent Space')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.colorbar(label='Cluster')
    plt.show()

# Plot clusters for males
plot_clusters(latent_representations_male, clusters_male, KMeans(n_clusters=4, random_state=42).fit(latent_representations_male).cluster_centers_)

# Plot

# Plot clusters for females
plot_clusters(latent_representations_female, clusters_female, KMeans(n_clusters=4, random_state=42).fit(latent_representations_female).cluster_centers_)
