import numpy as np
import pandas as pd
import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# Setup
# =========================

load_dotenv()
client = OpenAI(api_key=os.getenv("OPEN_API_KEY"))

df = pd.read_csv("./data/core_dataset_with_index.csv")
df["tags_list"] = df["tags"].apply(
    lambda x: [t["name"] for t in eval(x)] if pd.notna(x) else []
)

embeddings = np.load("./data/embeddings.npy")

with open("./data/tags.json", "r", encoding="utf-8") as f:
    CLUSTERS = json.load(f)

# Build tag -> cluster map
tag_to_cluster = {}
for cluster_name, tag_list in CLUSTERS.items():
    for tag in tag_list:
        tag_to_cluster[tag.lower()] = cluster_name

# =========================
# Config
# =========================

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

STRONG_NEGATIVE_KEYWORDS = {"harem", "love triangle"}

# =========================
# Utilities
# =========================

def normalize(text):
    return str(text).lower().strip()


def get_query_embedding(query):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    return np.array(response.data[0].embedding, dtype=np.float32)


def get_query_insights(query):
    available_tags = ", ".join(tag_to_cluster.keys())

    prompt = f"""
User Query:
{query}

Available Tags:
[{available_tags}]

Return JSON:
{{
  "positive_tags": [],
  "negative_tags": []
}}

Rules:
- Only choose exact strings from Available Tags.
- Do not invent new tags.
- If none apply, return empty lists.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Extract structured tag intent."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0
    )

    return json.loads(response.choices[0].message.content)


# =========================
# Scoring Functions
# =========================

def compute_cluster_tag_score(query, candidate_tags):
    query_words = set(normalize(query).split())
    total = 0
    count = 0

    for tag in candidate_tags:
        tag_norm = normalize(tag)
        tag_words = set(tag_norm.split())
        overlap = tag_words & query_words

        if not overlap:
            continue

        cluster = tag_to_cluster.get(tag_norm)
        if cluster in HIGH_PRIORITY:
            base = 1.0
        elif cluster in MEDIUM_PRIORITY:
            base = 0.5
        else:
            continue

        total += base * (len(overlap) / len(tag_words))
        count += 1

    return total / count if count else 0.0


def compute_intent_score(candidate_tags, positive_tags, negative_tags):
    candidate_norm = {normalize(t) for t in candidate_tags}
    positive_norm = {normalize(t) for t in positive_tags}
    negative_norm = {normalize(t) for t in negative_tags}

    score = 0
    score += len(candidate_norm & positive_norm)
    score -= len(candidate_norm & negative_norm)

    return score


def strong_negative_hit(row, negative_tags):
    title = normalize(row.get("title_english") or row.get("title_romaji"))
    description = normalize(row.get("description"))

    negative_norm = {normalize(t) for t in negative_tags}

    # Tag-based
    if negative_norm & {normalize(t) for t in row["tags_list"]}:
        return True

    # Keyword-based
    for keyword in STRONG_NEGATIVE_KEYWORDS:
        if keyword in title or keyword in description:
            return True

    # Genre-based (if Romance and user said NOT romance)
    if "romance" in negative_norm:
        genres = normalize(str(row.get("genres", "")))
        if "romance" in genres:
            return True

    return False


# =========================
# Search
# =========================

def search(query, top_k=10):

    print(f"\nSearching for: {query}\n")

    insights = get_query_insights(query)
    positive_tags = insights.get("positive_tags", [])
    negative_tags = insights.get("negative_tags", [])

    query_vec = get_query_embedding(query)
    cosine_scores = cosine_similarity([query_vec], embeddings).flatten()

    top_idx = np.argsort(cosine_scores)[-300:][::-1]
    candidates = df.iloc[top_idx].copy()
    candidates["cosine"] = cosine_scores[top_idx]

    # Compute structured scores
    candidates["tag_score"] = candidates["tags_list"].apply(
        lambda tags: compute_cluster_tag_score(query, tags)
    )

    candidates["intent_score"] = candidates["tags_list"].apply(
        lambda tags: compute_intent_score(tags, positive_tags, negative_tags)
    )

    # Normalize structured components
    if candidates["tag_score"].max() > 0:
        candidates["tag_score"] /= candidates["tag_score"].max()

    if candidates["intent_score"].abs().max() > 0:
        candidates["intent_score"] /= candidates["intent_score"].abs().max()

    # Base score
    candidates["final_score"] = (
        0.60 * candidates["cosine"] +
        0.20 * candidates["tag_score"] +
        0.20 * candidates["intent_score"]
    )

    # Strong negative penalty (multiplicative)
    for idx, row in candidates.iterrows():
        if strong_negative_hit(row, negative_tags):
            candidates.at[idx, "final_score"] *= 0.65

    results = candidates.sort_values("final_score", ascending=False).head(top_k)

    # Output
    for i, (_, row) in enumerate(results.iterrows(), 1):
        title = row["title_english"] if pd.notna(row["title_english"]) else row["title_romaji"]
        print(f"{i}. {title} ({row['year']})")
        print(f"   Score: {row['final_score']:.4f}")
        print()

    return results