from pytorch_lightning import Trainer
from stl_text.datamodule import TextClassificationDataModule
from task import TextClassificationTask

if __name__ == "__main__":

    LR = 5  # learning rate
    NUM_CLASS = 4  # number of classes
    EMBED = 256  # embedding
    EPOCH = 3  # max epoch number

    data_module = TextClassificationDataModule()
    task = TextClassificationTask(len(data_module.vocab), EMBED, NUM_CLASS, LR)
    trainer = Trainer(gpus=1, max_epochs=EPOCH, progress_bar_refresh_rate=40)
    trainer.fit(task, data_module)

    # run test set
    result = trainer.test()
    print("The accuracy for test set is: {:4.3f}".format(result[0]['test_acc']))
