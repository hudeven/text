import torchtext
import io
from pytorch_lightning import Trainer
from stl_text.datamodule import TextClassificationDataModule
from task import TextClassificationTask
import datasets as ds
from torchtext.utils import download_from_url, unicode_csv_reader

# Those are some utils to preprocess data. Will move those torchtext.
def create_data_from_csv(data_path):
    with io.open(data_path, encoding="utf8") as f:
        reader = unicode_csv_reader(f)
        for row in reader:
            yield (int(row[0]), ' '.join(row[1:]))


def convert_to_arrow(file_path, raw_data):
    """ Write labels and texts into HF dataset"""
    labels, texts = zip(*raw_data)
    ds.Dataset.from_dict(
        {
            "labels": labels,
            "texts": texts
        }).save_to_disk(file_path)


if __name__ == "__main__":

    LR = 5  # learning rate
    NUM_CLASS = 4  # number of classes
    EMBED = 256  # embedding
    EPOCH = 3  # max epoch number

    # Generate raw text data
    base_url = 'https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/'
    train_filepath = download_from_url(base_url + 'train.csv')
    test_filepath = download_from_url(base_url + 'test.csv')
    raw_train_data = list(create_data_from_csv(train_filepath))
    raw_test_data = list(create_data_from_csv(test_filepath))
    convert_to_arrow('train_arrow', raw_train_data)
    convert_to_arrow('test_arrow', raw_test_data)

    # Generate DataModule and Task
    data_module = TextClassificationDataModule(train_arrow_path='train_arrow',
                                               test_arrow_path='test_arrow')
    task = TextClassificationTask(len(data_module.vocab), EMBED, NUM_CLASS, LR)
    trainer = Trainer(gpus=1, max_epochs=EPOCH, progress_bar_refresh_rate=40)
    trainer.fit(task, data_module)

    # run test set
    result = trainer.test()
    print("The accuracy for test set is: {:4.3f}".format(result[0]['test_acc']))
