import pandas as pd

train_df = pd.read_csv("/project/lt200324-optmul/pluem/version_3/train_dataset.csv")

nan_count = train_df.isna().sum()
print(nan_count)

unique_df = (
    train_df.dropna(subset=["text"])
    .drop_duplicates(subset="text", keep="first")
    .reset_index(drop=True)
)
print(f"Original: {len(train_df)}")
print(f"Remove NaN: {nan_count}")
print(f"Cleansed: {len(unique_df)}")

unique_df.to_csv("/project/lt200324-optmul/pluem/version_3/train_dataset_final.csv")