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
