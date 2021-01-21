import os
import unittest
import tempfile

import pandas as pd
from stl_text.dataframe import DataFrame


class DataFrameTest(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = os.path.join(os.path.dirname(__file__), "data")
        self.test_file = os.path.join(self.test_dir, "text_label.tsv")
        self.test_dict = {
            "text": ["hello world", "attention is all you need!", "foo bar"],
            "label": ["Positive", "Negative", "Positive"],
        }

    def test_save_load_file(self):
        df = DataFrame.from_dict(self.test_dict)
        _, file_path = tempfile.mkstemp(prefix="stl_text_")
        df.save_to_file(file_path)
        df2 = df.from_file(file_path)
        self.assertEqual(len(df2), len(df))

    def test_from_dict(self):
        df = DataFrame.from_dict(self.test_dict)
        self.assertEqual(len(df), 3)

    def test_from_csv(self):
        df = DataFrame.from_csv(self.test_file, fieldnames=["text", "label"])
        self.assertEqual(len(df), 20)

    def test_from_pandas(self):
        pandas_df = pd.DataFrame.from_dict(self.test_dict)
        df = DataFrame.from_pandas(pandas_df)
        self.assertEqual(len(df), 3)

    def test_get_item(self):
        expected_len = 20
        df = DataFrame.from_csv(self.test_file, fieldnames=["text", "label"])
        self.assertEqual(len(df["text"]), expected_len)
        self.assertEqual(len(df["label"]), expected_len)
        self.assertEqual(df[0], {'label': '1', 'text': "it 's a charming and often affecting journey . "})
        self.assertEqual(df[-1], {'label': '0',
                                  'text': 'in its best moments , resembles a bad high school production of grease , without benefit of song . '})

    def test_iter(self):
        df = DataFrame.from_csv(self.test_file, fieldnames=["text", "label"])
        count = 0
        for _ in df:
            count += 1
        self.assertEqual(count, 20)

    def test_map(self):
        df = DataFrame.from_csv(self.test_file, fieldnames=["text", "label"])
        # tokenization
        df2 = df.map(lambda x: {"tokens": x["text"].split()})
        self.assertEqual(len(df2["tokens"]), len(df["text"]))
        self.assertEqual(len(df2["tokens"][0]), 9)

        # generate sequence length
        df3 = df2.map(lambda x: {"seq_len": len(x)}, input_columns=["tokens"])
        self.assertEqual(len(df3["seq_len"]), len(df2["tokens"]))
        self.assertEqual(df3["seq_len"][0], 9)

