from dataclasses import dataclass
import logging
from typing import List, Optional

import datasets
import torch
from pytorch_lightning import LightningDataModule
from torch import nn
from torch.nn.utils.rnn import pad_sequence


class TranslationDataModule(LightningDataModule):
    def __init__(
        self,
        source_data: datasets.arrow_dataset.Dataset,
        target_data: datasets.arrow_dataset.Dataset,
        source_vocab: "vocab",
        target_vocab: "vocab",
        batch_size_sequences: Optional[int] = 16,
        batch_size_tokens: Optional[int] = None,
        source_text_transform: Optional[nn.Module] = None,
        target_text_transform: Optional[nn.Module] = None,
        num_dataloader_workers: int = 1,
        num_setup_workers: int = 4,
    ):
        super().__init__()
        self.source_data = source_data
        self.target_data = target_data
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        self.batch_size_sequences = batch_size_sequences
        self.batch_size_tokens = batch_size_tokens
        self.source_text_transform = source_text_transform
        self.target_text_transform = target_text_transform
        self.num_dataloader_workers = num_dataloader_workers
        self.num_setup_workers = num_setup_workers

        # TODO should torchtext vocab define pad, bos, eos, etc?
        # TODO otherwise take these in as input arguments?
        self.source_pad_idx = 0
        self.target_bos_idx = 0
        self.target_pad_idx = -100

        assert self.source_pad_idx >= 0, \
            f"source_pad_idx must be >= 0 for nn.Embedding ({self.source_pad_idx})"
        assert self.target_bos_idx >= 0, \
            f"target_bos_idx must be >= 0 for nn.Embedding ({self.target_bos_idx})"

        if batch_size_tokens is not None:
            raise NotImplementedError

        if batch_size_sequences is None and batch_size_tokens is None:
            raise ValueError

        self.datasets = {}

    def setup(self, stage):
        for split in ("train", "valid", "test"):
            src = self.source_data[split].map(
                function=lambda x: {"token_ids": self.source_text_transform(x)},
                input_columns="text",
                num_proc=self.num_setup_workers,
            )
            src.set_format(type="torch", columns=["token_ids"])
            tgt = self.target_data[split].map(
                function=lambda x: {"token_ids": self.target_text_transform(x)},
                input_columns="text",
                num_proc=self.num_setup_workers,
            )
            tgt.set_format(type="torch", columns=["token_ids"])

            self.datasets[split] = DictDataset({
                "source": src,
                "target": tgt,
            })

    def forward(self, source_text_batch: List[str], target_prefix_batch: Optional[List[str]] = None) -> torch.Tensor:
        if target_prefix_batch is not None:
            raise NotImplementedError
        data = [
            {"source": {"token_ids": self.source_text_transform(text)}}
            for text in source_text_batch
        ]
        return torch.utils.data.DataLoader(
            data,
            shuffle=False,
            # TODO allow batch size to be configured here? do we even want batching?
            batch_size=self.batch_size_sequences,
            collate_fn=self.collate,
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.datasets["train"],
            shuffle=True,
            batch_size=self.batch_size_sequences,
            num_workers=self.num_dataloader_workers,
            collate_fn=self.collate,
        )

    def valid_dataloader(self):
        return torch.utils.data.DataLoader(
            self.self.datasets["valid"],
            shuffle=True,
            batch_size=self.batch_size_sequences,
            num_workers=self.num_dataloader_workers,
            collate_fn=self.collate,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.datasets["test"],
            shuffle=False,
            batch_size=self.batch_size_sequences,
            num_workers=self.num_dataloader_workers,
            collate_fn=self.collate,
        )

    def collate(self, batch):
        source_toks = pad_sequence(
            [x["source"]["token_ids"] for x in batch],
            batch_first=True,
            padding_value=self.source_pad_idx,
        )

        if "target" in batch[0]:
            target_bos_tensor = source_toks.new([self.target_bos_idx])
            target_pad_tensor = source_toks.new([self.target_pad_idx])
            teacher_forcing_toks = pad_sequence(
                [torch.cat([target_bos_tensor, x["target"]["token_ids"]]) for x in batch],
                batch_first=True,
                padding_value=self.target_pad_idx,
            )
            target_toks = pad_sequence(
                [torch.cat([x["target"]["token_ids"], target_pad_tensor]) for x in batch],
                batch_first=True,
                padding_value=self.target_pad_idx,
            )
        else:
            teacher_forcing_toks = None
            target_toks = None

        return {
            "source_token_ids": source_toks,
            "teacher_forcing_token_ids": teacher_forcing_toks,
            "target_token_ids": target_toks,
        }


class DictDataset(torch.utils.data.Dataset):

    def __init__(self, dictionary):
        super().__init__()
        self.dictionary = dictionary

        # TODO add more checks
        ref = next(iter(dictionary.values()))
        for v in dictionary.values():
            assert len(v) == len(ref)
        self._len = len(ref)

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        return {k: v[index] for k, v in self.dictionary.items()}
