import os
import tempfile

import json
import argparse
import jsonlines
import torch
from torch.optim import AdamW
from pytorch_lightning import Trainer, LightningModule
from stl_text.ops.utils.arrow import convert_reader_to_arrow
from stl_text.datamodule import ContrastivePretrainingDataModule
from stl_text.models import RobertaModel
from copy import deepcopy


def train(data_path:str, max_epochs: int, gpus: int, fast_dev_run: bool = False):
    # convert jsonl to arrow format (only required for the first time)
    for split in ("train.json", "valid.json", "test.json"):
        split_path = os.path.join(data_path, split)
        print(split_path)
        with jsonlines.open(split_path) as reader:
            convert_reader_to_arrow(
                reader, output_path=os.path.splitext(split_path)[0])

    # setup datamodule
    datamodule = DenseRetrieverDataModule(
        data_path=data_path, batch_size=4, drop_last=True)
    datamodule.setup("fit")

    # Model for query encoding
    query_model = RobertaModel(
        vocab_size=5000,
        embedding_dim=512,
        num_attention_heads=1,
        num_encoder_layers=1,
        output_dropout=0.4,
        out_dim=20,
    )
    context_model = deepcopy(query_model)

    optimizer = AdamW(model.parameters(), lr=0.01)

    task = DenseRetrieverTask(
        query_model=query_model,
        context_model=context_model,
        datamodule=datamodule,
        optimizer=optimizer
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


def select_data_for_debug(data_path="../DPR/data/squad1/data/retriever/squad1-dev.json", 
                          out_path="examples/dpr/data/squad1_tiny/data/retriever", 
                          num_items=[10, 10, 10]):
    """
    This generates jsonl files from 
    the input data from https://github.com/facebookresearch/DPR/blob/master/data/download_data.py
    ```
    # Get some data
    # Clone DPR - https://github.com/facebookresearch/DPR
    cd ../
    git clone git@github.com:facebookresearch/DPR.git
    cd DPR
    pip install .
    python data/download_data.py --resource data.retriever.squad1-dev --output_dir data/squad1
    ```
    """
    with open(data_path) as f_in:
        data = json.load(f_in)
        data_selected = [x for x in data if len(x["positive_ctxs"]) > 0 and len(
            x["negative_ctxs"]) > 0 and len(x["hard_negative_ctxs"]) > 0]
        print("Total items with non-empty contexts:{}".format(len(data_selected)))

        if isinstance(num_items, int):
            num_items = 3 * [num_items]

        for split_id, split in enumerate(["train.jsonl", "valid.jsonl", "test.jsonl"]):
            split_path = os.path.join(out_path, split)
            print(split_path)
            with open(split_path, mode="w") as f_out:
                for item in data_selected[:num_items[split_id]]:
                    f_out.write(json.dumps(item))
                    f_out.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DPR retriever')
    parser.add_argument('--max_epochs', type=int, default=5,
                        help='num epochs (default=5)')
    parser.add_argument('--gpus', type=int, default=0, help='num of gpus')
    parser.add_argument('--fast_dev_run', action="store_true",
                        help='fast train with a iteration')
    parser.add_argument(
        '--data', type=str, default="examples/dpr/data/squad1_tiny/data/retriever", help='The path to the data dir')

    args = parser.parse_args()

    max_epochs = args.max_epochs
    gpus = args.gpus
    fast_dev_run = args.fast_dev_run
    data_path = args.data_path

    task = train(data_path, max_epochs, gpus, fast_dev_run=fast_dev_run)
