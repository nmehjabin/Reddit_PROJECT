import pandas as pd

df = pd.read_csv("cybersecurity_posts_clean_basic.csv")
df["text"] = df["text"].fillna("").astype(str)

# burnout_regex = r"(?<!/)\bburnout\b(?!/)"



# burnout_regex = r"(?<!/)\bburnout\b(?!/)"

mask = df["text"].str.contains(
    "burnout",
    case=False,
    regex=True
)

df_burnout = df[mask].copy()

print("Total posts:", len(df))
print("Posts mentioning burnout (clean):", mask.sum())

# df_burnout.to_csv("posts_burnout_only.csv", index=False)


df_burnout.to_csv("posts_burnout_only_inputfile.csv", index=False)

#this is for checking the number from the input csv file only
# print("Total posts:", len(df))
# print("Posts matching burnout regex:", mask.sum())
