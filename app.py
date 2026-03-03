import streamlit as st
import pandas as pd
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import os
import dotenv
import gdown

os.makedirs("data", exist_ok=True)

url = "https://drive.google.com/uc?id=1A5stE3WbJrz59A_tOLPHrqeis6WY4uH8"
file_path = "./data/embeddings.npy"

if not os.path.exists(file_path):
    st.write("Downloading embeddings...")
    gdown.download(url, file_path, quiet=False)

url_tags = "https://drive.google.com/uc?id=1idjtvXhrESMGPUNTaAwMaOsacxMwDhxT"
file_path_tags = "./data/tags.json"
if not os.path.exists(file_path_tags):
    st.write("Downloading tags...")
    gdown.download(url_tags, file_path_tags, quiet=False)

url_csv = "https://drive.google.com/uc?id=1dWzFITd2bsGOvFYoA0ltbNvZCtzpa_kd"
file_path_csv = "./data/core_dataset_with_index.csv"
if not os.path.exists(file_path_csv):
    st.write("Downloading dataset...")
    gdown.download(url_csv, file_path_csv, quiet=False)

from scripts.recommend import search


# Load env
try:
    dotenv.load_dotenv()
except:
    pass

st.title("🎯 Comic Recommender (Hybrid Search)")

query = st.text_area(
    "📝 Enter description",
    placeholder="e.g. strategic academy protagonist hiding true power regression"
)

top_k = st.selectbox(
    "📌 Number of recommendations",
    options=[3, 5, 10, 15, 20],
    index=2
)

if st.button("🔍 Recommend"):

    if not query.strip():
        st.warning("Please enter a description.")
        st.stop()

    with st.spinner("🔎 Searching..."):
        try:
            results = search(query, top_k=top_k)
        except Exception as e:
            st.error(f"Search failed: {e}")
            st.stop()


    if results.empty:
        st.warning("No results found.")
        st.stop()

    for _, row in results.iterrows():

        # Clean title logic
        title = row["title_english"] if pd.notna(row["title_english"]) else row["title_romaji"]

        # Rating display
        if pd.notna(row["averageScore"]):
            rating_text = f"{row['averageScore']}/100"
        else:
            rating_text = "Unrated"

        st.subheader(f"{title} ({rating_text})")
        st.caption(f"Year: {int(row['year']) if pd.notna(row['year']) else 'Unknown'}")

        st.write(row["description"][:500] + "...")

        image_url = row["coverImage"]

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
        except Exception:
            st.warning("⚠️ Image load failed.")

        st.markdown("---")