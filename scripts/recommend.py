import numpy as np
import pandas as pd
import json
from openai import OpenAI
import os
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity


load_dotenv()
client = OpenAI(api_key=os.getenv("OPEN_API_KEY"))

# Load dataset
df = pd.read_csv("./data/core_dataset_with_index.csv")

# Load embeddings
embeddings = np.load("./data/embeddings.npy")

# Load clustered tags
with open("./data/tags.json", "r", encoding="utf-8") as f:
    CLUSTERS = json.load(f)


HIGH_PRIORITY = {
    "Fantasy Core",
    "Isekai & Time",
    "Action & Combat",
    "Sci-Fi & Tech",
    "Crime & Underworld",
    "Psychological & Dark Themes",
    "Historical & Cultural",
    "Setting & Environment"
}

MEDIUM_PRIORITY = {
    "School & Education",
    "Work & Professions",
    "Sports & Competition",
    "Arts & Entertainment",
    "Slice of Life & Healing"
}

# Build tag -> cluster map
tag_to_cluster = {}
for cluster_name, tag_list in CLUSTERS.items():
    for tag in tag_list:
        tag_to_cluster[tag.lower()] = cluster_name


def normalize(text):
    return str(text).lower().strip()

def get_query_embedding(query):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    return np.array(response.data[0].embedding, dtype=np.float32)

def compute_tag_overlap_score(query, candidate_tags):
    query_words = set(normalize(query).split())
    
    total_score = 0
    match_count = 0

    for tag in candidate_tags:
        tag_norm = normalize(tag)
        tag_words = set(tag_norm.split())

        overlap = tag_words & query_words
        
        if overlap:
            cluster = tag_to_cluster.get(tag_norm)

            # Base weight from cluster priority
            if cluster in HIGH_PRIORITY:
                base_weight = 1.0
            elif cluster in MEDIUM_PRIORITY:
                base_weight = 0.5
            else:
                continue  # ignore low-priority clusters

            # Partial match strength
            match_ratio = len(overlap) / len(tag_words)

            total_score += base_weight * match_ratio
            match_count += 1

    if match_count == 0:
        return 0.0

    # Normalize by number of matched tags
    normalized_score = total_score / match_count

    return normalized_score

df["tags_list"] = df["tags"].apply(lambda x: [t["name"] for t in eval(x)] if pd.notna(x) else [])


def search(query, top_k=10):
    print(f"\n🔎 Searching for: {query}\n")

    # Step 1 — Query embedding
    query_vec = get_query_embedding(query)

    # Step 2 — Cosine similarity with all entries
    cosine_scores = cosine_similarity(
        [query_vec], embeddings
    ).flatten()

    # Step 3 — Take top 300 candidates
    top_300_idx = np.argsort(cosine_scores)[-300:][::-1]

    candidates = df.iloc[top_300_idx].copy()
    candidates["cosine"] = cosine_scores[top_300_idx]

    # Step 4 — Cluster-aware re-ranking
    tag_scores = []
    for _, row in candidates.iterrows():
        tag_score = compute_tag_overlap_score(query, row["tags_list"])
        tag_scores.append(tag_score)

    candidates["tag_score"] = tag_scores

    # Normalize tag score
    if candidates["tag_score"].max() > 0:
        candidates["tag_score"] = candidates["tag_score"] / candidates["tag_score"].max()

    # Final score
    candidates["final_score"] = (
        0.75 * candidates["cosine"] +
        0.25 * candidates["tag_score"]
    )

    # Sort final results
    results = candidates.sort_values("final_score", ascending=False).head(top_k)

    # Display
    for i, (_, row) in enumerate(results.iterrows(), 1):
        title = row["title_english"] if pd.notna(row["title_english"]) else row["title_romaji"]
        print(f"{i}. {title} ({row['year']})")
        print(f"   Score: {row['final_score']:.4f}")
        print()

    return results

# -----------------------------
# 5️⃣ Example Usage
# -----------------------------
if __name__ == "__main__":
    search("query")