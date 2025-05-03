from datasets import load_dataset
from pythainlp.tokenize import word_tokenize
import os
import re
import random
from collections import Counter
import multiprocessing

# Constants
ROW_PER_BATCH = 1000000  # Number of dataset rows per batch
DATASET_PATH = "/project/lt200324-optmul/pluem/hg_datasets/thaigov-corpus"
CONFIG_NAME = None
OUTPUT_PATH = "/project/lt200324-optmul/pluem/version_2/dataset/thaigov"
COLUMN_NAME = "context"
FILE_NAME = "thaigov"
IS_THAI = True
NO_PIPE = True
MAX_TEXT_LENGTH = 500000  # Maximum text length for processing
THRESHOLD_LENGTH = 16  # Maximum allowable length for each token
NUM_PROC = multiprocessing.cpu_count()


def is_thai_or_english(sentence):
    pattern = re.compile(r"^[\u0E00-\u0E7F\u0020-\u007E]*$")
    return bool(pattern.match(sentence))


def is_valid_token(token):
    return (
        len(token) > 0
        and "http" not in token
        and "://" not in token
        and is_thai_or_english(token)
    )


def contains_alphabet(sentence):
    pattern = r"[a-zA-Z\u0E00-\u0E7F0-9\u0E50-\u0E59]"
    return re.search(pattern, sentence) is not None


def remove_quotes(sentence):
    if sentence.startswith('"') and sentence.endswith('"'):
        return sentence
    return re.sub(r'^"|"$', "", sentence)


def clean_punctuation(sentence):
    sentence = (
        sentence.replace("๏", "")
        .replace("&nbsp;", "")
        .replace("<unk>", "")
        .replace("= =", "")
    )
    sentence = sentence.replace("|", "") if NO_PIPE else sentence
    sentence = re.sub(r"={2,}", "", sentence)
    sentence = re.sub(r"(?<=\S)={2,}(?=\S)", " = ", sentence)
    sentence = re.sub(r'"{2,}', '"', sentence)
    sentence = re.sub(r"^!+", "", sentence).lstrip()
    sentence = re.sub(r"([!?.,=*@():;])\1{4,}", "", sentence)
    sentence = re.sub(
        r"([!?.,=*@():;])\1{3,}", lambda m: m.group(1) * random.randint(1, 2), sentence
    )
    sentence = re.sub(r"\s+", " ", sentence)
    return "" if re.match(r"^[!?.,=*@():;]+\s*$", sentence) else sentence


def process_token(token):
    if re.fullmatch(r"[^\w\u0E00-\u0E7F]+", token):
        symbol_types = re.findall(r"[!?.,=*@():;]", token)
        if symbol_types:
            least_common_symbol = Counter(symbol_types).most_common()[-1][0]
            return least_common_symbol * token.count(least_common_symbol)
    return token


def split_long_token(token, threshold_length):
    words = token.split()
    lines = []
    current_line = []
    current_length = 0

    for word in words:
        if current_length + len(word) + (1 if current_line else 0) <= threshold_length:
            current_line.append(word)
            current_length += len(word) + (1 if current_line else 0)
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]
            current_length = len(word)

    if current_line:
        lines.append(" ".join(current_line))

    return lines


def split_text(tokens, threshold_length=THRESHOLD_LENGTH):
    """Splits tokens into text blocks based on the threshold length and cleans unnecessary punctuation."""
    output = []
    sentence = ""

    for item in tokens:
        if not is_valid_token(item):
            continue

        item = process_token(item)

        if len(item) > threshold_length:
            if sentence:
                cleaned_sentence = clean_punctuation(sentence).rstrip()
                if cleaned_sentence and contains_alphabet(cleaned_sentence):
                    cleaned_sentence = remove_quotes(cleaned_sentence).strip()
                    output.append(cleaned_sentence)
                sentence = ""

            cleaned_item = clean_punctuation(item).rstrip()
            if cleaned_item and contains_alphabet(cleaned_item):
                output.extend(split_long_token(cleaned_item, threshold_length))

        elif len(sentence + item) > threshold_length:
            cleaned_sentence = clean_punctuation(sentence).rstrip()
            if cleaned_sentence and contains_alphabet(cleaned_sentence):
                cleaned_sentence = remove_quotes(cleaned_sentence).strip()
                output.append(cleaned_sentence)
            sentence = item.strip() if IS_THAI else item.strip() + " "
        else:
            sentence = (
                clean_punctuation(sentence).lstrip() + item
                if IS_THAI
                else clean_punctuation(sentence).lstrip() + item + " "
            )

    cleaned_sentence = clean_punctuation(sentence).rstrip()
    if cleaned_sentence and contains_alphabet(cleaned_sentence):
        cleaned_sentence = remove_quotes(cleaned_sentence).strip()
        output.append(cleaned_sentence)

    return output


def process_text(example):
    """Tokenizes and processes text, splitting it into chunks if it is below a length threshold."""
    text = example[COLUMN_NAME].replace("\n", " ")
    if len(text) > MAX_TEXT_LENGTH:
        example["chunked_text"] = []
    else:
        tokenized_text = (
            word_tokenize(text, engine="newmm") if IS_THAI else text.split()
        )
        example["chunked_text"] = split_text(tokenized_text)
    return example


def remove_duplicates(original_list):
    """Removes duplicate entries from the list."""
    seen = set()
    result = []
    for item in original_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def is_symbol_heavy(text):
    """Determines if the proportion of symbols in the text is higher than 1/3."""
    total_length = len(text)
    symbol_length = len(re.findall(r"[^\w\s]", text)) + len(re.findall(r"\s", text))
    return total_length > 0 and symbol_length / total_length > 1 / 3


def contains_thai(text):
    """Checks if the text contains Thai characters."""
    return bool(re.search(r"[\u0E00-\u0E7F]", text))


def thai_proportion(text):
    """Calculates the proportion of Thai characters in the text."""
    total_length = len(text)
    thai_length = len(re.findall(r"[\u0E00-\u0E7F]", text))
    return thai_length / total_length if total_length > 0 else 0


def filter_list(data):
    """Filters the list based on the presence of symbols and Thai character proportion."""
    if IS_THAI:
        return [
            item
            for item in data
            if not is_symbol_heavy(item)
            and (
                not re.search(r"[^\s]", item)
                or (contains_thai(item) and thai_proportion(item) >= 1 / 3)
            )
        ]
    else:
        return [item for item in data if not is_symbol_heavy(item)]


def process_chunk(start_index, end_index, dataset, num_proc=4):
    """Processes a chunk of the dataset by tokenizing and cleaning text."""
    subset_dataset = dataset["train"].select(range(start_index, end_index))
    processed_dataset = subset_dataset.map(process_text, num_proc=num_proc)
    chunked_text_lists = processed_dataset["chunked_text"]
    return [item for sublist in chunked_text_lists for item in sublist]


def main():
    """Main function to process the dataset and save cleaned text."""
    dataset = load_dataset(DATASET_PATH, CONFIG_NAME)
    all_text_list = []
    total_rows = len(dataset["train"])

    for start_index in range(0, total_rows, ROW_PER_BATCH):
        end_index = min(start_index + ROW_PER_BATCH, total_rows)
        print(f"Processing rows {start_index} to {end_index}")

        chunk_text_list = process_chunk(start_index, end_index, dataset, NUM_PROC)
        all_text_list.extend(chunk_text_list)

    # Remove duplicates and filter the processed text
    text_list_cleansed = remove_duplicates(all_text_list)
    text_list_cleansed = filter_list(text_list_cleansed)

    # Ensure the directory exists
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # Write cleaned text to a file using join for efficiency
    with open(f"{OUTPUT_PATH}/{FILE_NAME}.txt", "w", encoding="utf-8") as file:
        file.write("\n".join(line.strip() for line in text_list_cleansed))

    print(f"Finished processing. Total items: {len(text_list_cleansed)}")


if __name__ == "__main__":
    main()
