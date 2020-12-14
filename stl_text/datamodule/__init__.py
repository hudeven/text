from .doc_classification import ConcatPairDocClassificationDataModule, DocClassificationDataModule
from .translation import TranslationDataModule
from .contrastive_pretraining import ContrastivePretrainingDataModule

# As nightly version of torchtext is tricky to setup correctly
# We removed QuestionAnswerDataModule and TextClassificationDataModule to make dependency to torchtext optional

__ALL__ = [
    "ContrastivePretrainingDataModule",
    "ConcatPairDocClassificationDataModule",
    "DocClassificationDataModule",
    "TranslationDataModule",
]
