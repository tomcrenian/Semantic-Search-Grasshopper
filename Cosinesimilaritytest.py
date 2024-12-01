import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re

# Load the CSV file
#enter csv path under variable csv_path
df = pd.read_csv(csv_path)

# Convert embeddings from strings to lists of floats
df['embedding'] = df['embedding'].apply(lambda x: np.array(eval(x)))

# Convert the 'text' column to lowercase for easier case-insensitive matching
df['text'] = df['text'].str.lower()

# Function to get the top k similar words using cosine similarity
def find_similar_words(word, df, k=3):
    # Convert the input word to lowercase
    word = word.lower()
    
    # Get the embedding for the given word
    word_embedding = df.loc[df['text'] == word, 'embedding'].values
    if len(word_embedding) == 0:
        # If the word is not found, return an empty string or a default value
        return "N/A"
    
    word_embedding = word_embedding[0].reshape(1, -1)
    # Calculate cosine similarities between the given word and all other embeddings
    similarities = cosine_similarity(word_embedding, np.stack(df['embedding'].values))[0]
    
    # Add similarities to the dataframe
    df['similarity'] = similarities
    
    # Sort by similarity and get the top k results (excluding the word itself if present)
    top_k = df[df['text'] != word].nlargest(k, 'similarity')
    
    # Return the list of similar words as a comma-separated string
    return ', '.join(top_k['text'].values)

# Function to parse the text prompt into a dictionary
def parse_prompt(prompt_string):
    pattern = r"(\w+): ([^;]+)"
    matches = re.findall(pattern, prompt_string)
    return {key.strip(): value.strip() for key, value in matches}

# The input prompt as a text string
prompt_string = "Name: Joist; Category: Structural; Subcategory: Beam; Form: Rectangular; Material: Steel; U-value: 10; Durability: Durable."

# Parse the text prompt into a dictionary
prompt = parse_prompt(prompt_string)

# Fields that we want to expand with similar words
fields_to_expand = ["Name", "Form", "Material"]

# Create the expanded prompt by incorporating similar words where applicable
expanded_prompt = {}
for key, value in prompt.items():
    if key in fields_to_expand:
        expanded_prompt[key] = f"{value} ({find_similar_words(value, df)})"
    else:
        expanded_prompt[key] = value

# Create the formatted prompt string
formatted_prompt = "; ".join([f"{key}: {value}" for key, value in expanded_prompt.items()]) + "."

# Print the formatted prompt
print(formatted_prompt)
