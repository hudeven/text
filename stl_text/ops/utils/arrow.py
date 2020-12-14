import csv
import logging
import os
from collections import defaultdict
from typing import Tuple

import datasets as ds

logger = logging.getLogger(__name__)


def convert_csv_to_arrow(input_path: str, output_path: str = None, fieldnames: Tuple[str] = ("text", "label")):
    """ Write labels and texts into HF dataset"""
    columnar_data = defaultdict(list)
    for row in read_data_from_csv(input_path, fieldnames):
        for column, value in row.items():
            columnar_data[column].append(value)

    if not output_path:
        output_path = os.path.splitext(input_path)[0]
    logger.warning(f"converted to arrow and saved to {output_path}")
    return ds.Dataset.from_dict(columnar_data).save_to_disk(output_path)


def read_data_from_csv(data_path: str, fieldnames: Tuple[str] = ("text", "label")):
    with open(data_path, encoding="utf-8") as csv_file:
        yield from csv.DictReader(
            csv_file, quotechar='"', delimiter="\t", quoting=csv.QUOTE_ALL, skipinitialspace=True,
            fieldnames=fieldnames
        )
