import os
import time
from collections import defaultdict

import datasets
from tqdm import tqdm
from stl_text.ops.utils.arrow import convert_csv_to_arrow
from stl_text.ops.tokenizers import WhitespaceTokenizer
from stl_text.ops.transforms import LabelTransform
from stl_text.ops.utils.arrow import read_data_from_csv


def main():
    # convert csv to arrow format (only required for the first time)
    data_path = "./glue_sst2_large"
    split = "train_100M"
    csv_path = os.path.join(data_path, split + ".tsv")
    convert_csv_to_arrow(csv_path)

    text_transform = WhitespaceTokenizer(trainable=False, speed=1000)
    label_transform = LabelTransform(["0", "1"])

    # Mock data processing in PyText
    start_time = time.time()
    tensorizers = defaultdict(list)
    for row in tqdm(list(read_data_from_csv(csv_path, fieldnames=("text", "label")))):
        tensorizers["tokens"].append(text_transform(row["text"]))
        tensorizers["labels"].append(label_transform(row["label"]))
    print(f'processed: {len(tensorizers["tokens"])} examples')
    print("--- %s seconds ---" % (time.time() - start_time))

    # Arrow single process
    start_time = time.time()
    ds = datasets.Dataset.load_from_disk(os.path.join(data_path, split))  # raw dataset
    ds = ds.map(function=lambda x: {'label_id': label_transform(x)}, input_columns='label', load_from_cache_file=False)
    ds = ds.map(function=lambda x: {'token_ids': text_transform(x)}, input_columns='text', load_from_cache_file=False)
    print(f"processed: {len(ds)} examples")
    print("--- %s seconds ---" % (time.time() - start_time))

    # Arrow 8 process
    start_time = time.time()
    ds = datasets.Dataset.load_from_disk(os.path.join(data_path, split))  # raw dataset
    ds = ds.map(function=lambda x: {'label_id': label_transform(x)}, input_columns='label', load_from_cache_file=False, num_proc=8)
    ds = ds.map(function=lambda x: {'token_ids': text_transform(x)}, input_columns='text', load_from_cache_file=False, num_proc=8)
    print(f"processed: {len(ds)} examples")
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()
