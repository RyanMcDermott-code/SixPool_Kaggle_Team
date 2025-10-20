import pandas as pd
import ollama
import json
import os

df = pd.read_csv('data/train.csv')

for index, row in df.iterrows():
    prompt = f"""You are a judge evaluating which AI response a user preferred.

DATA:
- Model A: {row['model_a']}
- Model B: {row['model_b']}
- Prompt: {row['prompt']}
- Response A: {row['response_a']}
- Response B: {row['response_b']}

TASK:
Analyze both responses and predict which one the user preferred based on quality, helpfulness, accuracy, and relevance.

RESPONSE FORMAT:
You must respond with ONLY valid JSON. No additional text, explanations, or markdown.
Set exactly ONE field to 1 and the others to 0.

- Set "ai_winner_model_a" to 1 if Response A is better
- Set "ai_winner_model_b" to 1 if Response B is better  
- Set "ai_winner_tie" to 1 if both responses are equally good

Example output:
{{"ai_winner_model_a": 1, "ai_winner_model_b": 0, "ai_winner_tie": 0}}

Your JSON response:"""
   
    response = ollama.chat(
        model='gemma3:1b',
        messages=[{'role': 'user', 'content': prompt}],
        format='json'
    )
    print(response['message']['content'])