import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/train.csv')

print('total', len(df))
print('winner a: ', (df['winner_model_a'] == 1).sum())
print('winner b: ', (df['winner_model_b'] == 1).sum())
print('tie: ', (df['winner_tie'] == 1).sum())


# Get all unique models from both columns
models_a = df['model_a'].unique()
models_b = df['model_b'].unique()
all_models = list(set(list(models_a) + list(models_b)))

# Count wins for each model
win_counts = {}
for model in all_models:
    # Count wins when model is in model_a column
    wins_as_a = ((df['model_a'] == model) & (df['winner_model_a'] == 1)).sum()
    # Count wins when model is in model_b column
    wins_as_b = ((df['model_b'] == model) & (df['winner_model_b'] == 1)).sum()
    # Total wins
    win_counts[model] = wins_as_a + wins_as_b

# Convert to DataFrame for easier plotting
wins_df = pd.DataFrame(list(win_counts.items()), columns=['model', 'wins'])
wins_df = wins_df.sort_values('wins', ascending=False)

print(wins_df)

# Check max length for each column
text_columns = ['prompt', 'response_a', 'response_b']

print("Max lengths by column:")
for col in text_columns:
    max_len = df[col].str.len().max()
    print(f"{col}: {max_len} characters")

# Find the overall maximum across all three columns
max_overall = df[text_columns].apply(lambda x: x.str.len()).max().max()
print(f"\nOverall max length across all three columns: {max_overall} characters")

# Optional: Find which column and row has the overall maximum
max_lengths = df[text_columns].apply(lambda x: x.str.len())
max_col = max_lengths.max().idxmax()
max_row = max_lengths[max_col].idxmax()

print(f"\nLongest text is in column '{max_col}' at row {max_row}")
print(f"Length: {max_lengths.loc[max_row, max_col]} characters")
print(f"\nPreview of longest text:")
print(df.loc[max_row, max_col][:200] + "...")  # Show first 200 chars