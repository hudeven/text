import os
import tempfile

import torch
from torch.optim import AdamW
from pytorch_lightning import Trainer
from stl_text.ops.utils.arrow import convert_csv_to_arrow
from stl_text.datamodule import DocClassificationDataModule
from stl_text.models import RobertaModel
from task import DocClassificationTask


def main():
    # convert csv to arrow format (only required for the first time)
    data_path = "./glue_sst2_tiny"
    for split in ("train.tsv", "valid.tsv", "test.tsv"):
        split_path = os.path.join(data_path, split)
        convert_csv_to_arrow(split_path)

    # setup datamodule
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
    trainer = Trainer(max_epochs=5, fast_dev_run=True)
    trainer.fit(task, datamodule=datamodule)

    # test model
    trainer.test(task, datamodule=datamodule)

    # export task(transform + model) to TorchScript
    export_path = "/tmp/doc_classification_task.pt1"
    task.to_torchscript(export_path)

    # deploy task to server and inference
    with open(export_path, "rb") as f:
        ts_module = torch.load(f)
        print(ts_module(text_batch=["hello world", "attention is all your need!"]))


if __name__ == "__main__":
    main()
