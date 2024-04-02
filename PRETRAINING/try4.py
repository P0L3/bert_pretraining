import pandas as pd

df = pd.read_csv("DATASET/BATCHED/ED4RE_MSL512_ASL50_S3551425", nrows=100)

for row in df["Text"]:
    print(row)
    print(20*"----")