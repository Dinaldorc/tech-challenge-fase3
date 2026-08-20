import pandas as pd

df = pd.read_parquet("data/gold/FT_MACHINE_LEARNING.parquet")
print("shape:", df.shape)
print("anos:", sorted(df["ANO_REFERENCIA"].unique()))
print("target:", df["TARGET"].value_counts(normalize=True))
print("nulos top 10:")
print((df.isna().mean() * 100).sort_values(ascending=False).head(10))
print(df.sample(5, random_state=1).to_string())