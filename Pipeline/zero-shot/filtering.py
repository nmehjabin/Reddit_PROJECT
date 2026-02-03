import pandas as pd

df = pd.read_csv('cybersecurity_reddit_topics_labeled2.csv')
topic_0_rows = df[df['topic'] == 0]

# Save to new file
topic_0_rows.to_csv('topic_0_only.csv', index=False)
print(f"Saved {len(topic_0_rows)} rows with topic=0")
print(f"Total rows in original file: {len(df)}")