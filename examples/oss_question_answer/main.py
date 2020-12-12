import torch
from pytorch_lightning import Trainer
import datasets as ds
from torchtext.utils import download_from_url
from torchtext.experimental.datasets.raw import SQuAD1
from model import BertModel, QuestionAnswerModel
from task import QuestionAnswerTask
from stl_text.datamodule.question_answer import QuestionAnswerDataModule


def convert_to_arrow(raw_data, arrow_filepath):
    """ Write context/question/answers/ans_pos into HF dataset"""
    context, question, answers, ans_pos = zip(*raw_data)
    answers = [item[0] for item in answers]
    ans_pos_string = []
    for idx in range(len(ans_pos)):
        pos = ans_pos[idx][0]
        context_str = context[idx]
        ans_pos_string.append(context_str[:pos])
    ds.Dataset.from_dict(
        {
            "context": context,
            "question": question,
            "answers": answers,
            "ans_pos": ans_pos_string
        }).save_to_disk(arrow_filepath)


if __name__ == "__main__":
    # Hyperparameters
    EPOCH = 3  # epoch
    LR = 0.5  # learning rate
    BATCH_SIZE = 8  # batch size for training

    # Generate data module
    # It should be noted that this example is to demonstrate the idea on a pretrained model
    # We use only ~1% data to fine-tune the model.
    train, dev = SQuAD1()
    raw_train = list(train)[:1024]
    raw_dev = list(dev)[:128]
    convert_to_arrow(raw_train, "train_arrow")
    convert_to_arrow(raw_dev, "dev_arrow")

    base_url = 'https://pytorch.s3.amazonaws.com/models/text/torchtext_bert_example/'
    vocab_path = download_from_url(base_url + 'bert_vocab.txt')
    data_module = QuestionAnswerDataModule(train_arrow_path='train_arrow',
                                           dev_arrow_path='dev_arrow',
                                           vocab_filepath=vocab_path, batch_size=BATCH_SIZE)

    # Load pretrained model and generate task
    # default parameters from the pretrained model
    vocab_size, emsize, nhead, nhid, nlayers, dropout = 99230, 768, 12, 3072, 12, 0.2
    pretrained_bert = BertModel(vocab_size, emsize, nhead, nhid, nlayers, dropout)
    pretrained_model_path = download_from_url(base_url + 'ns_bert.pt')
    pretrained_bert.load_state_dict(torch.load(pretrained_model_path, map_location='cpu'))
    qa_model = QuestionAnswerModel(pretrained_bert)

    task = QuestionAnswerTask(qa_model, LR)
    trainer = Trainer(gpus=0, max_epochs=EPOCH, progress_bar_refresh_rate=40, fast_dev_run=True)
    trainer.fit(task, data_module)
