import pandas as pd, ast, re

df = pd.read_csv("data.csv")
df.dropna(subset=['description', 'tags'], inplace=True)
df = df[~df['description'].str.contains("no description", case=False)]
df = df[~df['description'].str.contains("this entry currently", case=False)]
df['tags'] = df['tags'].apply(ast.literal_eval)

banned_tags = {
    'BL', 'Yaoi', 'Shounen-ai', 'GL', 'Yuri', 'Shojo',
    'Smut', 'Sexual Abuse', 'Explicit Sex', 'Incest',
    'Omegaverse', 'Self-Harm', 'Cannibalism', 'BDSM'
}
df = df[~df['tags'].apply(lambda tags: any(tag in banned_tags for tag in tags))].reset_index(drop=True)

# Clean text
def clean_text(text):
    text = text.lower()
    return re.sub(r'[^\w\s]', '', text)

df['description'] = df['description'].apply(clean_text)

# Optionally combine tags into description for richer input
df['combined'] = df.apply(lambda row: f"Tags: {', '.join(row['tags'])}. Description: {row['description']}", axis=1)

df.to_csv("clean_data.csv", index=False)