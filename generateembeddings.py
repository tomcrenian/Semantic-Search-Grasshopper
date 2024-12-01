from openai import OpenAI
import pandas as pd
import numpy as np
import time

# Initialize the OpenAI client, add your key here
client = OpenAI(api_key="###################################################")

# Load list of words or phrases from a text file
with open(r'C:\Users\tomcr\Downloads\WECD-master\WECD-master\word_vectors\unique_names.txt') as file:
    text_list = [line.strip() for line in file]

# Function to get embedding for a batch of text inputs
def get_batch_embeddings(text_batch, model="text-embedding-3-small"):
    # Ensure that text does not contain newlines
    text_batch = [text.replace("\n", " ") for text in text_batch]
    response = client.embeddings.create(input=text_batch, model=model, dimensions=256)
    return [res.embedding for res in response.data]

# Function to get embeddings for a list of texts using batching
def generate_embeddings(text_list, batch_size=100, model="text-embedding-3-large"):
    embeddings = []
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i + batch_size]
        try:
            # Get embeddings for the current batch
            batch_embeddings = get_batch_embeddings(batch, model=model)
            embeddings.extend(batch_embeddings)
        except Exception as e:
            print(f"Error generating embeddings for batch starting at index {i}: {str(e)}")
            # If an error occurs, fill the batch with None values
            embeddings.extend([None] * len(batch))
        time.sleep(1)  # Add a delay to avoid rate limits
    return embeddings

# Generate embeddings for the list in batches
embeddings = generate_embeddings(text_list, batch_size=1000)  # Adjust batch size based on your API rate limits

# Save the embeddings to a CSV file for future use
embeddings_df = pd.DataFrame({'text': text_list, 'embedding': embeddings})
embeddings_df.to_csv('C://Users//tomcr//Documents//Projects//embeddings.csv', index=False)

