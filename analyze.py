import pandas as pd

# Load the dataset
df = pd.read_csv("C:\\Users\\Isha\\IdeaProjects\\untitled\\Python\\Projects\\Movie-Sentiment-Analysis\\IMDB Dataset.csv")

# See the first 5 rows
print(df.head())

# Get the dimensions of the dataset
print(f"\n--- Dataset Shape ---")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# Check for any missing values
print("\n--- Missing Values ---")
print(df.isnull().sum())

# Check the distribution of sentiments
print("\n--- Sentiment Distribution ---")
print(df['sentiment'].value_counts())