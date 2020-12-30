# DPR Retriever 

In this example we implement the retriever model of the Dense Passage Retriever (DPR) paper. Paper and original code:
- DPR repo: https://github.com/facebookresearch/DPR/
- DPR paper: https://arxiv.org/abs/2004.04906
- STL Text repo: https://github.com/hudeven/text

## About the code
We implement a simple DPR retriever training. 
The current directory contains a tiny data in the .examples/dpr/data folder that can be used to validate that the model is training (run with few iterations, see that loss is decreasing). The input data format is jsonl with the original data format (from DPR repo download script).
The example encoder model is a RobertaModel with a tiny size in order to enable faster debugging iterations.

## Data
The current setting has a tiny set with 10 examples.

## Get more data
To obtain the original data using https://github.com/facebookresearch/DPR/blob/master/data/download_data.py
```
# Get data
# Clone DPR - https://github.com/facebookresearch/DPR
cd ../
git clone git@github.com:facebookresearch/DPR.git
cd DPR
pip install .

# see the available resource keys:
python data/download_data.py 

# download the data using
python data/download_data.py --resource data.retriever.nq-dev --output_dir data/nq
python data/download_data.py --resource data.retriever.nq-train --output_dir data/nq
#python data/download_data.py --resource data.retriever.squad1-dev --output_dir data/squad1

# Select data
ORIG_DIR=../DPR/data/nq3/data/retriever/
OUT_DIR=examples/dpr/data/nq3_small/data/retriever/
PYTHONPATH=. python examples/dpr/main.py --sel_data --sel_raw_path=${ORIG_DIR}/nq-dev.json --sel_out_path=${OUT_DIR}/train.jsonl --sel_num_items=500
PYTHONPATH=. python examples/dpr/main.py --sel_data --sel_raw_path=${ORIG_DIR}/nq-dev.json --sel_out_path=${OUT_DIR}/valid.jsonl --sel_num_items=100
PYTHONPATH=. python examples/dpr/main.py --sel_data --sel_raw_path=${ORIG_DIR}/nq-dev.json --sel_out_path=${OUT_DIR}/test.jsonl --sel_num_items=100
```

## Run the training
```
DATA_DIR=examples/dpr/data/nq3_tiny/data/retriever/
PYTHONPATH=. python examples/dpr/main.py --max_epochs=10 --data_path=${DATA_DIR}
```
    