import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('historical_games/games_orig.csv')
columns_to_remove = ['Game', 'ECO', 'White', 'PlyCount','Black', 'White RD', 'Black RD', 'WhiteIsComp', 'BlackIsComp', 'TimeControl', 'Date', 'Time', 'White Clock', 'Black Clock', 'Commentaries']
df = df.drop(columns=columns_to_remove)

# Filter out games with a sum of Elo ratings less than 3500
df = df[df['White Elo'] >= 2000]
df = df[df['Black Elo'] >= 2000]

df = df[df['Moves'].str.split().str.len() / 2 >= 10]

# Save the modified DataFrame back to a CSV file
df.to_csv('historical_games/games_clean.csv', index=False)