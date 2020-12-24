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


class DPRRetrieverDataModule(LightningDataModule):
    def __init__(self, data_path: str, 
                vocab_path: Optional[str] = None, 
                batch_size: int = 32,
                train_max_positive: int = 10,
                train_max_negative: int = 10,
                train_ctxs_random_sample: bool = True, 
                drop_last: bool = False,
                num_proc_in_map: int = 1, 
                distributed: bool = False, 
                load_from_cache_file: bool = True):
        super().__init__()
        self.data_path = data_path
        self.vocab_path = vocab_path
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_proc_in_map = num_proc_in_map
        self.distributed = distributed
        self.load_from_cache_file = load_from_cache_file

        self.train_max_positive = train_max_positive
        self.train_max_negative = train_max_negative
        self.train_ctxs_random_sample = train_ctxs_random_sample

        self.text_transform = None
        self.datasets = {}


    def setup(self, stage):
        self.text_transform = WhitespaceTokenizer(vocab_path=self.vocab_path)

        for split in ("train", "valid", "test"):
            dataset_split = ds.Dataset.load_from_disk(os.path.join(self.data_path, split))  # raw dataset
            dataset_split = dataset_split.map(function=lambda x: {'query_ids': self.text_transform(x)},
                                                            input_columns='question', num_proc=self.num_proc_in_map,
                                                            load_from_cache_file=self.load_from_cache_file)
            dataset_split = dataset_split.map(function=lambda ctxs: {'contexts_pos_ids': [self.text_transform(x["text"]) for x in ctxs]},
                                                            input_columns='positive_ctxs', num_proc=self.num_proc_in_map,
                                                            load_from_cache_file=self.load_from_cache_file)
            dataset_split = dataset_split.map(function=lambda ctxs: {'contexts_neg_ids': [self.text_transform(x["text"]) for x in ctxs]},
                                                input_columns='negative_ctxs', num_proc=self.num_proc_in_map,
                                                load_from_cache_file=self.load_from_cache_file)
            dataset_split = dataset_split.map(function=lambda x: {'query_seq_len': len(x)},
                                                            input_columns='query_ids', num_proc=self.num_proc_in_map,
                                                            load_from_cache_file=self.load_from_cache_file)
            dataset_split.set_format(type='torch', columns=['query_ids', 'query_seq_len', 
                                                            'contexts_pos_ids', 'contexts_neg_ids'])
            
            self.datasets[split] = dataset_split

    def forward(self, text):
        return self.text_transform(text)

    def train_dataloader(self):
        # sample data into `num_batches_in_page` sized pool. In each pool, sort examples by sequence length, batch them
        # with `batch_size` and shuffle batches
        train_dataset = self.datasets["train"]
        batch_sampler = PoolBatchSampler(train_dataset, batch_size=self.batch_size,
                                         drop_last=self.drop_last, key=lambda row: row["query_seq_len"])
        return torch.utils.data.DataLoader(train_dataset, batch_sampler=batch_sampler,
                                           num_workers=1,
                                           collate_fn=self.collate_train)

    def valid_dataloader(self):
        return torch.utils.data.DataLoader(self.self.datasets["valid"], shuffle=True, batch_size=self.batch_size,
                                           num_workers=1,
                                           collate_fn=self.collate_eval)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.datasets["test"], shuffle=False, batch_size=self.batch_size,
                                           num_workers=1,
                                           collate_fn=self.collate_eval)

    def collate_eval(self, batch):
        return self.collate(batch, False)

    def collate_train(self, batch):
        return self.collate(batch, True)

    def collate(self, batch, is_train):
        """
            Combines pos and neg contexts. Samples randomly limited number of pos/neg contexts if is_train is True.
        """
        for row in batch:
            # sample positive contexts
            contexts_pos_ids = row["contexts_pos_ids"]
            if is_train and self.train_max_positive > 0:
                if self.train_ctxs_random_sample:
                    contexts_pos_ids = random.sample(contexts_pos_ids, self.train_max_positive) 
                else:
                    contexts_pos_ids = contexts_pos_ids[:self.train_max_positive]
            
            # sample positive contexts
            contexts_neg_ids = row["contexts_neg_ids"]
            if is_train and self.train_max_negative > 0:
                if self.train_ctxs_random_sample:
                    contexts_neg_ids = random.sample(contexts_neg_ids, self.train_max_negative) 
                else:
                    contexts_neg_ids = contexts_neg_ids[:self.train_max_negative]
            
            row["contexts_ids"] = contexts_pos_ids + contexts_neg_ids
            row["contexts_is_pos"] = [1] * len(contexts_pos_ids) + [0] * len(contexts_neg_ids)

            row.pop("contexts_pos_ids")
            row.pop("contexts_neg_ids")

        return self._collate(batch, pad_columns=('query_ids',
                                                 'contexts_ids',
                                                 'contexts_is_pos'))

    # generic collate(), same as DocClassificationDataModule
    def _collate(self, batch, pad_columns):
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
