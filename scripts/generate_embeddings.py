import openai, numpy as np, time, math
import os, dotenv

# Load environment variables
dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAIKEY")  # Secure your key

def get_openai_embeddings(texts, model="text-embedding-3-small"):
    all_embeddings = []
    batch_size = 200
    total_batches = math.ceil(len(texts) / batch_size)

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            print(f"🔄 Batch {i//batch_size + 1}/{total_batches}")
            response = openai.embeddings.create(input=batch, model=model)
            all_embeddings.extend([e.embedding for e in response.data])
        except Exception as e:
            print("⚠️ Error:", e)
            time.sleep(5)
    
    print("✅ Embedding complete")
    return np.array(all_embeddings)

import pandas as pd
df = pd.read_csv("clean_data.csv")
texts = df["combined"].tolist()

embeddings = get_openai_embeddings(texts)
np.save("clean_embeddings.npy", embeddings)
