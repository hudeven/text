import os

import torch
from pytorch_lightning import Trainer
from stl_text.ops.utils.arrow import convert_csv_to_arrow
from stl_text.datamodule import DocClassificationDataModule
from task import XlmrDocClassificationTask


if __name__ == "__main__":

    # convert csv to arrow format (only required for the first time)
    data_path = "./glue_sst2_tiny"
    for split in ("train.tsv", "valid.tsv", "test.tsv"):
        split_path = os.path.join(data_path, split)
        convert_csv_to_arrow(split_path)

    # setup datamodule
    datamodule = DocClassificationDataModule(data_path=data_path, batch_size=8, drop_last=True)
    datamodule.setup("fit")

    # construct task
    task = XlmrDocClassificationTask(text_transform=datamodule.text_transform, num_class=2,
                                     lr=0.01)

    # train model
    trainer = Trainer(max_epochs=5, fast_dev_run=True)
    trainer.fit(task, datamodule=datamodule)

    # export to TorchScript
    export_path = "/tmp/doc_task.pt1"
    task.to_torchscript("/tmp/doc_task.pt1")
    with open(export_path, "rb") as f:
        ts_module = torch.load(f)
        print(ts_module(text_batch=["hello world", "attend is all your need!"]))
