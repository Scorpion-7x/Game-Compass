import pandas as pd
import os

print("Starting Data Pre-processing...")

# Defining our input and output file
input_file = 'data/games.csv'
output_file = 'data/cleaned_games.csv'

# Check if the file exists
if not os.path.exists(input_file):
    print(f"ERROR: Could not find the file '{input_file}'.")
    print("Please make sure you created a 'data' folder and put 'games.csv' inside it!")
else:
    # Loading the dataset
    print("Loading Raw Dataset... this might take a second...")
    df = pd.read_csv(input_file)
    print(f"Raw Games Loaded: {len(df):,}")

    # Selecting the columns we need for our engine
    columns_to_keep = [
        'AppID', 'Name', 'About the game', 'Header image',
        'Positive', 'Negative', 'Price', 'Release date'
    ]
    df = df[columns_to_keep]

    # Simplifying Columns names
    df = df.rename(columns={
        'AppID': 'app_id', 
        'Name': 'name', 
        'About the game': 'description',
        'Header image': 'image_url',
        'Positive': 'positive_reviews',
        'Negative': 'negative_reviews',
        'Price': 'price',
        'Release date': 'release_date'
    })

    # Dropping any games that are missing a name or description
    df = df.dropna(subset=['name', 'description'])

    # Filtering out ghost games
    df['total_reviews'] = df['positive_reviews'] + df['negative_reviews']

    # Filtering games with >20 reviews
    df = df[df['total_reviews'] >= 20]

    # Calculating positive ratio
    df['positive_ratio'] = df['positive_reviews'] / df['total_reviews']

    print(f"Cleaned Database Ready: {len(df):,} verified games.")

    # Obtaining the cleaned Dataset
    df.to_csv(output_file, index=False)
    print(f"Cleaned dataset is saved as :  '{output_file}'")