from .doc_classification import ConcatPairDocClassificationDataModule, DocClassificationDataModule
from .translation import TranslationDataModule

# As nightly version of torchtext is tricky to setup correctly
# We removed QuestionAnswerDataModule and TextClassificationDataModule to make dependency to torchtext optional

__ALL__ = [
    "ConcatPairDocClassificationDataModule",
    "DocClassificationDataModule",
    "TranslationDataModule",
]
