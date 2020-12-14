import os
import random
from collections import defaultdict
from typing import Optional

import torch
import datasets as ds
from pytorch_lightning import LightningDataModule
from stl_text.ops.tokenizers import WhitespaceTokenizer
from stl_text.ops.transforms import LabelTransform
from torch.nn.utils.rnn import pad_sequence
from stl_text.ops.samplers import PoolBatchSampler
from .doc_classification import DocClassificationDataModule


class ContrastivePretrainingDataModule(LightningDataModule):
    def __init__(self, data_path: str = 'paraphrases', vocab_path: Optional[str] = None, batch_size: int = 32,
                 drop_last: bool = False,
                 num_proc_in_map: int = 1, distributed: bool = False, load_from_cache_file: bool = True):
        super().__init__()
        self.data_path = data_path
        self.vocab_path = vocab_path
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_proc_in_map = num_proc_in_map
        self.distributed = distributed
        self.load_from_cache_file = load_from_cache_file

        self.text_transform = None
        self.datasets = {}

    def setup(self, stage):
        self.text_transform = WhitespaceTokenizer(vocab_path=self.vocab_path)

        for split in ("train", "valid", "test"):
            self.datasets[split] = ds.Dataset.load_from_disk(os.path.join(self.data_path, split))  # raw dataset
            self.datasets[split] = self.datasets[split].map(function=lambda x: {'anchor_ids': self.text_transform(x)},
                                                            input_columns='anchor', num_proc=self.num_proc_in_map,
                                                            load_from_cache_file=self.load_from_cache_file)
            self.datasets[split] = self.datasets[split].map(function=lambda xs: {'all_positive_ids': [self.text_transform(x) for x in xs]},
                                                            input_columns='positives', num_proc=self.num_proc_in_map,
                                                            load_from_cache_file=self.load_from_cache_file)
            self.datasets[split] = self.datasets[split].map(function=lambda x: {'seq_len': len(x)},
                                                            input_columns='anchor_ids', num_proc=self.num_proc_in_map,
                                                            load_from_cache_file=self.load_from_cache_file)
            self.datasets[split].set_format(type='torch', columns=['anchor_ids', 'all_positive_ids', 'seq_len'])

    def forward(self, text):
        return self.text_transform(text)

    def train_dataloader(self):
        # sample data into `num_batches_in_page` sized pool. In each pool, sort examples by sequence length, batch them
        # with `batch_size` and shuffle batches
        batch_sampler = PoolBatchSampler(self.datasets["train"], batch_size=self.batch_size,
                                         drop_last=self.drop_last, key=lambda row: row["seq_len"])
        return torch.utils.data.DataLoader(self.datasets["train"], batch_sampler=batch_sampler,
                                           num_workers=1,
                                           collate_fn=self.collate)

    def valid_dataloader(self):
        return torch.utils.data.DataLoader(self.self.datasets["valid"], shuffle=True, batch_size=self.batch_size,
                                           num_workers=1,
                                           collate_fn=self.collate)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.datasets["test"], shuffle=False, batch_size=self.batch_size,
                                           num_workers=1,
                                           collate_fn=self.collate)

    def collate(self, batch):
        for row in batch:
            row["positive_ids"] = random.sample(row["all_positive_ids"], 1)[0]
            row.pop("all_positive_ids")
        return self._collate(batch, pad_columns=("anchor_ids", "positive_ids"))

    # generic collate(), same as DocClassificationDataModule
    def _collate(self, batch, pad_columns=("token_ids",)):
        columnar_data = defaultdict(list)
        for row in batch:
            for column, value in row.items():
                columnar_data[column].append(value)

        padded = {}
        for column, v in columnar_data.items():
            if pad_columns and column in pad_columns:
                padded[column] = pad_sequence(v, batch_first=True)
            else:
                padded[column] = torch.tensor(v, dtype=torch.long)
        return padded
