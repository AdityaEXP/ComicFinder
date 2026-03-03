import pandas as pd
import numpy as np
import math
import time
import os
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()
client = OpenAI(api_key=os.getenv("OPEN_API_KEY"))

MAX_CHARS = 12000 

df = pd.read_csv("./data/core_dataset.csv")


df["combined"] = (
    df["title_romaji"].fillna("") + ". " +
    df["title_english"].fillna("") + ". " +
    df["description"].fillna("")
)
df["combined"] = df["combined"].apply(
    lambda x: str(x)[:MAX_CHARS]
)

df = df.drop_duplicates(subset=["combined"]).reset_index(drop=True)

print("Total rows to embed:", len(df))

# Rough token estimate: words * 1.3
df["approx_tokens"] = df["combined"].apply(
    lambda x: int(len(str(x).split()) * 1.3)
)

total_tokens = df["approx_tokens"].sum()
cost_estimate = (total_tokens / 1_000_000) * 0.02  # $0.02 per 1M tokens

print(f"Approx total tokens: {total_tokens:,}")
print(f"Estimated embedding cost: ${cost_estimate:.4f}")

def get_openai_embeddings(texts, model="text-embedding-3-small"):
    all_embeddings = []
    batch_size = 200
    total_batches = math.ceil(len(texts) / batch_size)

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = start + batch_size
        batch = texts[start:end]

        print(f"🔄 Processing batch {batch_idx+1}/{total_batches}")

        retries = 3
        for attempt in range(retries):
            try:
                response = client.embeddings.create(
                    model=model,
                    input=batch
                )
                batch_embeddings = [e.embedding for e in response.data]
                all_embeddings.extend(batch_embeddings)
                break
            except Exception as e:
                print(f"⚠️ Attempt {attempt+1} failed:", e)
                time.sleep(5)
                if attempt == retries - 1:
                    raise e

    print("✅ Embedding complete")
    return np.array(all_embeddings, dtype=np.float32)

texts = df["combined"].tolist()
embeddings = get_openai_embeddings(texts)

np.save("./data/embeddings.npy", embeddings)
df.to_csv("./data/core_dataset_with_index.csv", index=False)

print("Saved:")
print(" - embeddings.npy")
print(" - core_dataset_with_index.csv")