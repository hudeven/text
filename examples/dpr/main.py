import os
import tempfile

import json
import argparse
import jsonlines
import torch
from torch.optim import AdamW
from pytorch_lightning import Trainer, LightningModule
from stl_text.ops.utils.arrow import convert_reader_to_arrow
from stl_text.datamodule.dpr import DPRRetrieverDataModule
from stl_text.models import RobertaModel
from copy import deepcopy
from task import DenseRetrieverTask


def train(data_path:str, max_epochs: int, gpus: int, batch_size:int, fast_dev_run: bool = False):
    # convert jsonl to arrow format (only required for the first time)
    for split in ("train.jsonl", "valid.jsonl", "test.jsonl"):
        split_path = os.path.join(data_path, split)
        print(split_path)
        with jsonlines.open(split_path) as reader:
            convert_reader_to_arrow(
                reader, output_path=os.path.splitext(split_path)[0])

    # setup datamodule
    datamodule = DPRRetrieverDataModule(
            data_path=data_path, 
            batch_size=batch_size, 
            drop_last=True,
            train_max_positive=1,
            train_max_negative=7,
            train_ctxs_random_sample=True,
            vocab_trainable=True 
    )
    datamodule.setup("fit")

    # Model for query encoding
    query_model = RobertaModel(
        vocab_size=20000,
        embedding_dim=512,
        num_attention_heads=1,
        num_encoder_layers=1,
        output_dropout=0.4,
        out_dim=20,
    )

    context_model = RobertaModel(
        vocab_size=20000,
        embedding_dim=512,
        num_attention_heads=1,
        num_encoder_layers=1,
        output_dropout=0.4,
        out_dim=20,
    )

    task = DenseRetrieverTask(
        query_model=query_model,
        context_model=context_model,
        datamodule=datamodule
    )

    # train model
    trainer = Trainer(
        max_epochs=max_epochs,
        gpus=gpus,
        fast_dev_run=fast_dev_run,
        accelerator=None, #"ddp" if gpus > 0 else None,
        replace_sampler_ddp=False
    )
    trainer.fit(task, datamodule=datamodule)

    # test model
    trainer.test(task, datamodule=datamodule)
    
    return task


def select_data_for_debug(original_path="../DPR/data/nq3/data/retriever/squad1-dev.json", 
                          out_path="examples/dpr/data/nq3_tiny/data/retriever/train.jsonl", 
                          num_items=10):
    """
    This generates jsonl files from 
    the input data from https://github.com/facebookresearch/DPR/blob/master/data/download_data.py
    ```
    # Get data
    # Clone DPR - https://github.com/facebookresearch/DPR
    cd ../
    git clone git@github.com:facebookresearch/DPR.git
    cd DPR
    pip install .
    python data/download_data.py --resource data.retriever.nq-dev --output_dir data/nq
    #python data/download_data.py --resource data.retriever.squad1-dev --output_dir data/squad1
    
    ```
    """
    with open(original_path) as f_in:
        data = json.load(f_in)
        # select only examples with both positive and negative contexts.
        data_selected = [x for x in data if len(x["positive_ctxs"]) > 0 and len(
            x["negative_ctxs"]) > 0 and len(x["hard_negative_ctxs"]) > 0]
        print("Total items with non-empty contexts:{}".format(len(data_selected)))

        out_dir = os.path.dirname(out_path)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            
        print(out_path)
        with open(out_path, mode="w") as f_out:
            for item in data_selected[:num_items]:
                f_out.write(json.dumps(item))
                f_out.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DPR retriever')
    parser.add_argument('--max_epochs', type=int, default=5,
                        help='num epochs (default=5)')
    parser.add_argument('--gpus', type=int, default=0, help='num of gpus')
    parser.add_argument('--fast_dev_run', action="store_true",
                        help='fast train with a iteration')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument(
        '--data_path', type=str, default="examples/dpr/data/nq3_tiny/data/retriever", help='The path to the data dir')

 
    # Select data
    parser.add_argument('--sel_data', action="store_true",
                        help='Select original data and convert to jsonl')
    parser.add_argument('--sel_raw_path', type=str, default="../DPR/data/nq3/data/retriever/nq-dev.json", help='Inpuit data dir for preparation')
    parser.add_argument('--sel_out_path', type=str, default="examples/dpr/data/nq3_tiny/data/retriever/train.jsonl", help='The path to the data dir')
    parser.add_argument('--sel_num_items', type=int, default=10, help='Number items to select')

    # Parse args
    args = parser.parse_args()

    max_epochs = args.max_epochs
    gpus = args.gpus
    fast_dev_run = args.fast_dev_run
    data_path = args.data_path
    batch_size = args.batch_size

    sel_data = args.sel_data
    if sel_data:
        select_data_for_debug(original_path=args.sel_raw_path, 
                          out_path=args.sel_out_path, 
                          num_items=args.sel_num_items)
    else:
        train(data_path, max_epochs, gpus, batch_size=batch_size, fast_dev_run=fast_dev_run)
