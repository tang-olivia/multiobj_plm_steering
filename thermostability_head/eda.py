import pandas as pd

df = pd.read_csv("cross-species.csv")

# drop NA meltPoints
df = df[df["meltPoint"] != "NA"]
df["meltPoint"] = pd.to_numeric(df["meltPoint"], errors="coerce")
df = df.dropna(subset=["meltPoint"])

# median Tm per protein
median_tm = df.groupby("Protein_ID")["meltPoint"].median().reset_index()
median_tm.columns = ["Protein_ID", "median_Tm"]

print(f"Proteins after aggregation: {len(median_tm)}")
print(median_tm.head())


print(df.shape)
print(df["meltPoint"].isna().sum())  # or == "NA"
print(df["Protein_ID"].nunique())    # unique proteins BEFORE dropping NA
median_tm.to_csv("median_tm.csv", index=False)