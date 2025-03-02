import pandas as pd

# Load the dataset
df = pd.read_csv("Crop_recommendation.csv")

# Display dataset information
print("Dataset Columns:", df.columns.tolist())
print("\nMissing Values:\n", df.isnull().sum())
print("\nDataset Summary:\n", df.describe())

# Check first few rows
print("\nFirst 5 Rows:\n", df.head())
