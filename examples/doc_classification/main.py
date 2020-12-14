import os
import tempfile

import argparse
import torch
from torch.optim import AdamW
from pytorch_lightning import Trainer, LightningModule
from stl_text.ops.utils.arrow import convert_csv_to_arrow
from stl_text.datamodule import ConcatPairDocClassificationDataModule, DocClassificationDataModule
from stl_text.models import RobertaModel
from task import DocClassificationTask


def train(max_epochs: int, gpus: int, fast_dev_run: bool = False, pair_classification: bool = False):
    # convert csv to arrow format (only required for the first time)
    if pair_classification:
        data_path = "./glue_mrpc_tiny"
        fieldnames = ("label", "id1", "id2", "text1", "text2")
    else:
        data_path = "./glue_sst2_tiny"
        fieldnames = ("text", "label")
    for split in ("train.tsv", "valid.tsv", "test.tsv"):
        split_path = os.path.join(data_path, split)
        convert_csv_to_arrow(split_path, fieldnames=fieldnames)

    # setup datamodule
    if pair_classification:
        datamodule = ConcatPairDocClassificationDataModule(data_path=data_path, batch_size=8, drop_last=True)
    else:
        datamodule = DocClassificationDataModule(data_path=data_path, batch_size=8, drop_last=True)
    datamodule.setup("fit")

    # build task
    model = RobertaModel(
        vocab_size=1000,
        embedding_dim=1000,
        num_attention_heads=1,
        num_encoder_layers=1,
        output_dropout=0.4,
        out_dim=2,
    )
    optimizer = AdamW(model.parameters(), lr=0.01)
    task = DocClassificationTask(
        datamodule=datamodule,
        model=model,
        optimizer=optimizer,
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


def export_and_inference(task: LightningModule):
    # export task(transform + model) to TorchScript
    export_path = "/tmp/doc_classification_task.pt1"
    task.to_torchscript(export_path)

    # deploy task to server and inference
    with open(export_path, "rb") as f:
        ts_module = torch.jit.load(f)

    text_batch = ["hello world", "unify pytext, fairseq, torchtext", "attention is all your need!"]
    print(f"Inference: \ninput: {text_batch}")
    logits = ts_module(text_batch=text_batch)
    print(f"output: logits = {logits}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a doc classification model')
    parser.add_argument('--max_epochs', type=int, default=5, help='num epochs (default=5)')
    parser.add_argument('--gpus', type=int, default=0, help='num of gpus')
    parser.add_argument('--fast_dev_run', type=bool, default=False, help='fast train with a iteration')
    parser.add_argument('--pair_classification', action="store_true", help='perform pair classification')
    args = parser.parse_args()

    max_epochs = args.max_epochs
    gpus = args.gpus
    fast_dev_run = args.fast_dev_run
    pair_classification = args.pair_classification

    task = train(max_epochs, gpus, fast_dev_run=fast_dev_run, pair_classification=pair_classification)
    export_and_inference(task)
