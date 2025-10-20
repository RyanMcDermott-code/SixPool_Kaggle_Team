import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('data/train.csv')

# Get all unique models
models_a = df['model_a'].unique()
models_b = df['model_b'].unique()
all_models = list(set(list(models_a) + list(models_b)))

# Count wins, losses, ties, and total appearances for each model
model_stats = {}
for model in all_models:
    # When model is model_a
    wins_as_a = ((df['model_a'] == model) & (df['winner_model_a'] == 1)).sum()
    losses_as_a = ((df['model_a'] == model) & (df['winner_model_b'] == 1)).sum()
    ties_as_a = ((df['model_a'] == model) & (df['winner_tie'] == 1)).sum()
    
    # When model is model_b
    wins_as_b = ((df['model_b'] == model) & (df['winner_model_b'] == 1)).sum()
    losses_as_b = ((df['model_b'] == model) & (df['winner_model_a'] == 1)).sum()
    ties_as_b = ((df['model_b'] == model) & (df['winner_tie'] == 1)).sum()
    
    # Totals
    total_wins = wins_as_a + wins_as_b
    total_losses = losses_as_a + losses_as_b
    total_ties = ties_as_a + ties_as_b
    total_appearances = total_wins + total_losses + total_ties
    
    # Calculate win rate
    win_rate = total_wins / total_appearances if total_appearances > 0 else 0
    
    model_stats[model] = {
        'wins': total_wins,
        'appearances': total_appearances,
        'win_rate': win_rate
    }

# Convert to DataFrame
stats_df = pd.DataFrame.from_dict(model_stats, orient='index')

# Create scatter plot
plt.figure(figsize=(12, 8))
plt.scatter(stats_df['appearances'], stats_df['win_rate'], s=100, alpha=0.6)

# Add labels for each point
for model, row in stats_df.iterrows():
    plt.annotate(model, (row['appearances'], row['win_rate']), 
                xytext=(5, 5), textcoords='offset points', fontsize=9)

plt.xlabel('Number of Appearances')
plt.ylabel('Win Rate')
plt.title('Model Win Rate vs Appearances')
plt.grid(True, alpha=0.3)
plt.ylim([0, 1])
plt.tight_layout()
plt.show()

print(stats_df.sort_values('win_rate', ascending=False))