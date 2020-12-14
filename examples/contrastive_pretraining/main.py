import os
import tempfile

import argparse
import jsonlines
import torch
from torch.optim import AdamW
from pytorch_lightning import Trainer, LightningModule
from stl_text.ops.utils.arrow import convert_reader_to_arrow
from stl_text.datamodule import ContrastivePretrainingDataModule
from stl_text.models import RobertaModel
from task import CertTask


def train(max_epochs: int, gpus: int, fast_dev_run: bool = False):
    # convert jsonl to arrow format (only required for the first time)
    data_path = "./paraphrases"
    for split in ("train.jsonl", "valid.jsonl", "test.jsonl"):
        split_path = os.path.join(data_path, split)
        print(split_path)
        with jsonlines.open(split_path) as reader:
            convert_reader_to_arrow(reader, output_path=os.path.splitext(split_path)[0])

    # setup datamodule
    datamodule = ContrastivePretrainingDataModule(data_path=data_path, batch_size=4, drop_last=True)
    datamodule.setup("fit")

    # build task
    model = RobertaModel(
        vocab_size=1000,
        embedding_dim=1000,
        num_attention_heads=1,
        num_encoder_layers=1,
        output_dropout=0.4,
        out_dim=20,
    )
    optimizer = AdamW(model.parameters(), lr=0.01)
    task = CertTask(
        datamodule=datamodule,
        model=model,
        optimizer=optimizer,
        embedding_dim=20,
        queue_size=4,
    )

    # train model
    trainer = Trainer(
        max_epochs=max_epochs,
        gpus=gpus,
        fast_dev_run=fast_dev_run,
        accelerator="ddp" if gpus > 0 else None,
        replace_sampler_ddp=False
    )
    trainer.fit(task, datamodule=datamodule)

    # test model
    trainer.test(task, datamodule=datamodule)
    return task


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a doc classification model')
    parser.add_argument('--max_epochs', type=int, default=5, help='num epochs (default=5)')
    parser.add_argument('--gpus', type=int, default=0, help='num of gpus')
    parser.add_argument('--fast_dev_run', action="store_true", help='fast train with a iteration')
    args = parser.parse_args()

    max_epochs = args.max_epochs
    gpus = args.gpus
    fast_dev_run = args.fast_dev_run

    task = train(max_epochs, gpus, fast_dev_run=fast_dev_run)
