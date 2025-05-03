import os
import re
import pandas as pd

BASE_PATH = "/project/lt200324-optmul/pluem/version_3/"


def extract_number(filename):
    """Extract numeric part from the filename."""
    match = re.search(r"(\d+)", filename)
    return int(match.group(0)) if match else 0


def list_files(directory, extension):
    """List all files in directory with given extension."""
    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(extension)
    ]


def get_image_paths(base_dir):
    """Get all image paths from subdirectories sorted numerically."""
    all_images = []
    for root, dirs, files in os.walk(base_dir):
        # Only consider files in the base directory
        if root == base_dir:
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                images = list_files(dir_path, ".jpg")
                all_images.extend(images)

    # Sort images based on the numeric part
    all_images.sort(key=lambda x: extract_number(os.path.basename(x)))
    return all_images


def get_texts_from_chunks(base_dir):
    """Read all text files from base directory and return combined text lines."""
    all_texts = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".txt"):
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    all_texts.extend(f.read().strip().split("\n"))
    return all_texts


def create_dataframe(image_paths, texts):
    """Create a DataFrame from image paths and texts."""
    if len(image_paths) != len(texts):
        print(len(image_paths), len(texts))
        raise ValueError(
            "The number of images does not match the number of text lines."
        )

    data = {"image_path": image_paths, "text": texts}

    return pd.DataFrame(data)


def process_category(category):
    """Process a single category of images and text files."""
    images_dir = os.path.join(BASE_PATH, "images", category)
    texts_dir = os.path.join(BASE_PATH, "dataset", category)

    image_paths = get_image_paths(images_dir)
    texts = get_texts_from_chunks(texts_dir)

    df = create_dataframe(image_paths, texts)
    df["category"] = category  # Add category column for identification
    return df


# Scan for all categories in the 'images' directory
image_base_dir = os.path.join(BASE_PATH, "images")
categories = [
    d
    for d in os.listdir(image_base_dir)
    if os.path.isdir(os.path.join(image_base_dir, d))
]

all_dfs = []

for category in categories:
    try:
        print(f"Processing category: {category}")
        df = process_category(category)
        all_dfs.append(df)
    except Exception as e:
        print(f"Error processing category {category}: {e}")

# Concatenate all DataFrames into one
combined_df = pd.concat(all_dfs, ignore_index=True)
print(len(combined_df))

unique_df = (
    combined_df.dropna(subset=["text"])
    .drop_duplicates(subset="text", keep="first")
    .reset_index(drop=True)
)
print(unique_df.head())
print(len(unique_df))

unique_df.to_csv(
    "/project/lt200324-optmul/pluem/version_3/train_dataset.csv", index=False
)
