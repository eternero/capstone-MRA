import pandas as pd

df = pd.read_csv('/Users/nico/Desktop/CIIC/CAPSTONE/essentia_demo/metadata.csv')

# Group by 'ALBUM' and 'ARTIST' by dropping duplicates
grouped_df = df[['ALBUM', 'ARTIST']].drop_duplicates()

# Save the grouped DataFrame to a new CSV file
grouped_df.to_csv('grouped_output.csv', index=False)
