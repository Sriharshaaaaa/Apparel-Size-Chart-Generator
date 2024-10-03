import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import tensorflow as tf
from tensorflow.keras import layers, models

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

# GAN Generator Model with even more complexity
def build_generator(latent_dim, n_features):
    model = models.Sequential()
    model.add(layers.Dense(512, activation='relu', input_dim=latent_dim))
    model.add(layers.Dropout(0.4))  # Dropout for regularization
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(n_features, activation='relu'))
    return model

# GAN Discriminator Model with increased complexity and regularization
def build_discriminator(n_features):
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_dim=n_features))
    model.add(layers.Dropout(0.4))  # Dropout for regularization
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

# Build and compile the GAN model with adjusted learning rates
def build_gan(generator, discriminator):
    discriminator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), loss='binary_crossentropy', metrics=['accuracy'])
    discriminator.trainable = False
    gan_input = layers.Input(shape=(latent_dim,))
    generated_data = generator(gan_input)
    gan_output = discriminator(generated_data)
    gan = models.Model(gan_input, gan_output)
    gan.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005), loss='binary_crossentropy')
    return gan

# Train the GAN
def train_gan(gan, generator, discriminator, data, epochs=10000, batch_size=32):
    half_batch = batch_size // 2

    for epoch in range(epochs):
        # Train discriminator with a batch of real data with label smoothing
        idx = np.random.randint(0, data.shape[0], half_batch)
        real_data = data[idx]
        real_labels = np.ones((half_batch, 1)) * 0.9
        d_loss_real = discriminator.train_on_batch(real_data, real_labels)

        # Train discriminator with a batch of fake data with label smoothing
        noise = np.random.normal(0, 1, (half_batch, latent_dim))
        fake_data = generator.predict(noise)
        fake_labels = np.zeros((half_batch, 1)) + 0.1
        d_loss_fake = discriminator.train_on_batch(fake_data, fake_labels)

        # Train generator via GAN
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        valid_y = np.ones((batch_size, 1))
        g_loss = gan.train_on_batch(noise, valid_y)

        # Calculate total discriminator loss
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Print losses at every epoch
        print(f"Epoch {epoch + 1}/{epochs} - D loss: {d_loss} - G loss: {g_loss}")

# Decoding one-hot encoded categorical features
def decode_one_hot(one_hot_encoded, encoder):
    decoded = []
    categories = encoder.categories_
    start_idx = 0
    for cat in categories:
        end_idx = start_idx + len(cat)
        one_hot_segment = one_hot_encoded[:, start_idx:end_idx]
        decoded_cat = np.argmax(one_hot_segment, axis=1)
        decoded.append([cat[i] for i in decoded_cat])
        start_idx = end_idx
    return np.array(decoded).T

# Decode synthetic data
def decode_synthetic_data(synthetic_data, scaler, encoder, numerical_features, categorical_features):
    decoded_data = scaler.inverse_transform(synthetic_data)
    num_features = decoded_data[:, :len(numerical_features)]
    cat_features = decoded_data[:, len(numerical_features):]
    decoded_cat_features = decode_one_hot(cat_features, encoder)
    return pd.DataFrame(np.hstack((num_features, decoded_cat_features)), columns=numerical_features + categorical_features)

# Preprocess male and female data
numerical_features_male = ['Bust/Chest', 'Height', 'Weight', 'Waist', 'Hips']
categorical_features_male = ['Body Shape Index']

numerical_features_female = ['Bust/Chest', 'Height', 'Weight', 'Waist', 'Hips']
categorical_features_female = ['Body Shape Index', 'Cup Size']

X_male_scaled, male_scaler, male_onehot_encoder = preprocess_data(male_data, numerical_features_male, categorical_features_male)
X_female_scaled, female_scaler, female_onehot_encoder = preprocess_data(female_data, numerical_features_female, categorical_features_female)

# Set parameters for GAN
latent_dim = 1000  # Further increased latent dimensionality
n_features_male = X_male_scaled.shape[1]
n_features_female = X_female_scaled.shape[1]

# Build GAN models for male data
generator_male = build_generator(latent_dim, n_features_male)
discriminator_male = build_discriminator(n_features_male)
gan_male = build_gan(generator_male, discriminator_male)

# Train GAN on male data
train_gan(gan_male, generator_male, discriminator_male, X_male_scaled, epochs=10000)

# Generate synthetic data for males
noise_male = np.random.normal(0, 1, (1000, latent_dim))
synthetic_data_male = generator_male.predict(noise_male)

# Decode male synthetic data
decoded_synthetic_male = decode_synthetic_data(synthetic_data_male, male_scaler, male_onehot_encoder, numerical_features_male, categorical_features_male)

# Combine real and synthetic male data
combined_male_data = pd.concat([pd.DataFrame(male_data, columns=numerical_features_male + categorical_features_male), decoded_synthetic_male])

# Save the combined male data to CSV
combined_male_data.to_csv('combined_male_data.csv', index=False)

# Build GAN models for female data
generator_female = build_generator(latent_dim, n_features_female)
discriminator_female = build_discriminator(n_features_female)
gan_female = build_gan(generator_female, discriminator_female)

# Train GAN on female data
train_gan(gan_female, generator_female, discriminator_female, X_female_scaled, epochs=10000)

# Generate synthetic data for females
noise_female = np.random.normal(0, 1, (1000, latent_dim))
synthetic_data_female = generator_female.predict(noise_female)

# Decode female synthetic data
decoded_synthetic_female = decode_synthetic_data(synthetic_data_female, female_scaler, female_onehot_encoder, numerical_features_female, categorical_features_female)

# Combine real and synthetic female data
combined_female_data = pd.concat([pd.DataFrame(female_data, columns=numerical_features_female + categorical_features_female), decoded_synthetic_female])

# Save the combined female data to CSV
combined_female_data.to_csv('combined_female_data.csv', index=False)

# Print samples of the generated data
print("Sample of Synthetic Male Data:")
print(decoded_synthetic_male.head())

print("\nSample of Synthetic Female Data:")
print(decoded_synthetic_female.head())

