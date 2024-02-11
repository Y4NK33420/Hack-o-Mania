import pandas as pd


df = pd.read_csv(r'D:\python projects\hack-o-mania 2024\master_comments.csv')

grouped = df.groupby('Author')['negative'].nunique()
authors_with_multiple_negatives = grouped[grouped > 1].index.tolist()
print(authors_with_multiple_negatives)