import pandas as pd
import numpy as np
import openai
import ast
import re
from sklearn.metrics.pairwise import cosine_similarity
import os, dotenv
# Load environment variables
dotenv.load_dotenv()

# ========== CONFIG ==========
OPENAI_API_KEY = os.getenv("OPENAIKEY")  # Replace with your actual key or use environment variable
EMBEDDING_MODEL = "text-embedding-3-small"
DATA_PATH = "data/clean_data.csv"
EMBEDDINGS_PATH = "data/clean_embeddings.npy"
TOP_K = 5
# ============================

# Load API Key
openai.api_key = OPENAI_API_KEY

# Clean user input
def clean_text(text):
    text = text.lower()
    return re.sub(r"[^\w\s]", "", text)

# Load data and embeddings
df = pd.read_csv(DATA_PATH)
df['tags'] = df['tags'].apply(ast.literal_eval)
df['description'] = df['description'].astype(str)
embeddings = np.load(EMBEDDINGS_PATH)

# Ensure combined field is used
if 'combined' not in df.columns:
    df['description'] = df['description'].apply(clean_text)
    df['combined'] = df.apply(lambda row: f"Tags: {', '.join(row['tags'])}. Description: {row['description']}", axis=1)

# Get user query
user_input = input("Enter a description to find similar manhwa (please do not include tags here): ")
tags_input = input("Enter tags (comma-separated, e.g. action, romance): ")
query = clean_text(user_input)
query_full = f"Description: {query}. Tags: {tags_input}"  # optionally, add default tag hints

# Embed query
try:
    print("\n🔍 Getting OpenAI embedding...")
    response = openai.embeddings.create(
        input=[query_full],
        model=EMBEDDING_MODEL
    )
    query_vector = np.array(response.data[0].embedding)
except Exception as e:
    print(f"❌ Failed to embed query: {e}")
    exit()

# Calculate similarity
scores = cosine_similarity([query_vector], embeddings).flatten()
top_indices = np.argsort(scores)[-TOP_K:][::-1]
recommendations = df.iloc[top_indices]

# Print results
print("\n🎯 Top Recommended Manhwa:\n")
for i, row in recommendations.iterrows():
    print(f"{row['title']} ({row['rating']}/5)")
    print(f"Genres: {', '.join(row['tags'])}")
    print(f"→ {row['description'][:200]}...\n")
