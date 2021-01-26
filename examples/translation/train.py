#!/usr/bin/env python3

import logging
import os
import tempfile
from itertools import chain
from typing import Dict, Iterable, Optional

import torch
from datasets import load_dataset
from pytorch_lightning import Trainer
from stl_text.datamodule import TranslationDataModule
from stl_text.ops.tokenizers import WhitespaceTokenizer
from torch import nn

from task import TranslationTask


logger = logging.getLogger(__name__)

def main(fast_dev_run=True):
    """
    WMT'14 cs-en translation
    """

    import pdb
    pdb.set_trace()
    logger.info("preparing WMT'14 cs-en data")
    wmt14 = prepare_wmt14_cs_en()

    train = "train" if not fast_dev_run else "validation"

    logger.info("build vocabs")
    source_text_transform = WhitespaceTokenizer()
    source_text_transform.build_vocab(map(lambda x:x['text'],wmt14["cs"][train]))
    target_text_transform = WhitespaceTokenizer()
    target_text_transform.build_vocab(map(lambda x:x['text'],wmt14["en"][train]))
    source_vocab = source_text_transform.vocab.get_stoi()
    target_vocab = target_text_transform.vocab.get_stoi()
    
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
        export_path = os.path.join(tmp, "translation_task.pt")
        task.to_torchscript(export_path)
        with open(export_path, "rb") as f:
            ts_module = torch.load(f)
            print(ts_module(source_text_batch=["hello world", "attend is all your need!"]))


class Model(nn.Module):
    def __init__(self, source_vocab_size, target_vocab_size):
        super().__init__()
        self.src_embed = nn.Embedding(source_vocab_size, 512)
        self.src_encode = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
        )
        self.tgt_embed = nn.Embedding(target_vocab_size, 512)
        self.tgt_encode = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
        )
        self.out = nn.Linear(512, target_vocab_size)

    def forward(self, batch: Dict[str, torch.Tensor], batch_idx: Optional[int]):
        x = self.src_embed(batch["source_token_ids"])
        x = self.src_encode(x)
        x = x.mean(dim=1, keepdim=True)  # mean pool over source tokens

        x = x + self.tgt_embed(batch["teacher_forcing_token_ids"])
        x = self.tgt_encode(x)
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
