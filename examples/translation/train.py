#!/usr/bin/env python3

import logging
import os
import tempfile
from itertools import chain
from typing import Iterable

import torch
from datasets import load_dataset
from pytorch_lightning import Trainer
from stl_text.datamodule.translation import TranslationDataModule
from stl_text.ops.tokenizers import WhitespaceTokenizer
from torch import nn

from task import TranslationTask

logger = logging.getLogger(__name__)


def build_vocab(data: Iterable, transform: nn.Module) -> "Vocab":
    tokens = chain.from_iterable(
        map(lambda x: transform(x["text"]), data)
    )
    list(tokens)
    return transform.vocab
    # tokens = chain.from_iterable(
    #    map(lambda x: transform.tokenize(x["text"]), data)
    # )
    # counts = Counter(tokens).most_common()
    # TODO cannot pickle 'torchtext._torchtext.Vocab' object
    # return vocab(OrderedDict(counts), unk_token="<unk>")


def main(fast_dev_run=True):
    """
    WMT'14 cs-en translation
    """

    logger.info("preparing WMT'14 cs-en data")
    wmt14 = prepare_wmt14_cs_en()

    train = "train" if not fast_dev_run else "validation"

    logger.info("build vocabs")
    source_text_transform = WhitespaceTokenizer(trainable=True)
    target_text_transform = WhitespaceTokenizer(trainable=True)
    source_vocab = build_vocab(wmt14["cs"][train], source_text_transform)
    target_vocab = build_vocab(wmt14["en"][train], target_text_transform)

    logger.info("init data module")
    datamodule = TranslationDataModule(
        source_data={
            "train": wmt14["cs"][train],
            "valid": wmt14["cs"]["validation"],
            "test": wmt14["cs"]["test"],
        },
        target_data={
            "train": wmt14["en"][train],
            "valid": wmt14["en"]["validation"],
            "test": wmt14["en"]["test"],
        },
        source_vocab=source_vocab,
        target_vocab=target_vocab,
        batch_size_sequences=16,
        source_text_transform=source_text_transform,
        target_text_transform=target_text_transform,
        num_setup_workers=10,
    )
    datamodule.setup("fit")
    print(f"######### {len(source_vocab)}, {len(target_vocab)}")
    model = Model(len(source_vocab), len(target_vocab))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

    logger.info("init task")
    task = TranslationTask(model=model, optimizer=optimizer, datamodule=datamodule)

    logger.info("trainer.fit")
    trainer = Trainer(fast_dev_run=fast_dev_run)
    trainer.fit(task, datamodule=datamodule)

    # export to TorchScript
    with tempfile.TemporaryDirectory() as tmp:
        export_path = os.path.join(tmp.name, "translation_task.pt")
        task.to_torchscript(export_path)
        with open(export_path, "rb") as f:
            ts_module = torch.load(f)
            print(ts_module(source_text_batch=["hello world", "attend is all your need!"]))


class Model(nn.Module):
    def __init__(self, source_vocab_size, target_vocab_size):
        super().__init__()
        self.embed = nn.Embedding(source_vocab_size, 512)
        self.fc1 = nn.Linear(512, 1024)
        self.out = nn.Linear(1024, target_vocab_size)

    def forward(self, batch, batch_idx):
        x = self.embed(batch["source_token_ids"])
        x = self.fc1(x)
        x = self.out(x)
        return (x,)


def prepare_wmt14_cs_en():
    wmt14 = load_dataset("wmt14", "cs-en")
    # convert from row-based indexing to column-based
    new_data = {}
    for lang in ("cs", "en"):
        new_data[lang] = {
            split: wmt14[split].map(
                function=lambda x: {"text": x["translation"][lang]},
                remove_columns=["translation"],
            )
            for split in ("train", "validation", "test")
        }
    return new_data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
