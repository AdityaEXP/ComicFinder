import streamlit as st
import pandas as pd, numpy as np, ast, re
from sklearn.metrics.pairwise import cosine_similarity
import openai, requests
from PIL import Image
from io import BytesIO
import os, dotenv, gdown

# downloading embeddings npy file
file_path = "data/clean_embeddings.npy"
if not os.path.exists(file_path):
    print("Downloading clean_embeddings.npy...")
    url = "https://drive.google.com/uc?id=1toZRablb8yCVhFrICdU1jQtmYh20mZ9P"
    gdown.download(url, file_path, quiet=False)

# Load environment variables
try:
    dotenv.load_dotenv()
    openai.api_key = os.getenv("OPENAIKEY")
except:
    import streamlit as st
    openai.api_key = st.secrets["OPENAIKEY"]
    
# Load data
df = pd.read_csv("data/clean_data.csv")
df['tags'] = df['tags'].apply(ast.literal_eval)
df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
embeddings = np.load("data/clean_embeddings.npy")

def clean_text(text):
    return re.sub(r"[^\w\s]", "", text.lower())

# --- UI ---
st.title("🎯 Manhwa Recommender")

desc = st.text_area("📝 Enter description (do not include tags)", placeholder="e.g. 'mc grows strong with magic'")
tags = st.text_input("🏷️ Tags (comma-separated)", value="romance, action")
top_k = st.selectbox("📌 Number of recommendations", options=[3, 5, 10, 15, 20], index=2)
ignore_nan_ratings = st.checkbox("🚫 Exclude manhwa with unknown ratings", value=True)

# Year range slider
min_year = int(df['year'].min())
max_year = int(df['year'].max())
year_range = st.slider("📅 Release year range", min_year, max_year, (min_year, max_year))

# --- Recommendation logic ---
if st.button("🔍 Recommend"):
    query = clean_text(desc)
    tag_str = ", ".join([t.strip() for t in tags.split(",")])
    full_input = f"Tags: {tag_str}. Description: {query}"

    with st.spinner("🔎 Embedding & searching..."):
        try:
            res = openai.embeddings.create(input=[full_input], model="text-embedding-3-small")
            query_vec = np.array(res.data[0].embedding)
            scores = cosine_similarity([query_vec], embeddings).flatten()

            # Sort by similarity
            sorted_indices = np.argsort(scores)[::-1]
            results = df.iloc[sorted_indices].copy()

            # Filter year range
            results = results[
                (results['year'] >= year_range[0]) & (results['year'] <= year_range[1])
            ]

            # Optional: exclude NaN ratings
            if ignore_nan_ratings:
                results = results[~results['rating'].isna()]

            # Take top K after filtering
            results = results.head(top_k)

        except Exception as e:
            st.error(f"❌ Failed: {e}")
            st.stop()

    # --- Display results ---
    for _, row in results.iterrows():
        st.subheader(f"{row['title']} ({row['rating']}/5)" if pd.notna(row['rating']) else f"{row['title']} (Unrated)")
        st.caption(f"Genres: {', '.join(row['tags'])}")
        st.write(f"{row['description']}...")

        image_url = row['cover']
        try:
            if pd.notnull(image_url) and isinstance(image_url, str):
                response = requests.get(image_url, headers={"User-Agent": "Mozilla/5.0"})
                if response.status_code == 200:
                    img = Image.open(BytesIO(response.content))
                    st.image(img, width=200)
                else:
                    st.warning("⚠️ Image couldn't be loaded.")
            else:
                st.warning("⚠️ No image available.")
        except Exception as e:
            st.error(f"Image load failed: {e}")
        st.markdown("---")
