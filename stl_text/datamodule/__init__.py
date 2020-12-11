from .doc_classification import DocClassificationDataModule
from .text_classification import TextClassificationDataModule
from .translation import TranslationDataModule
from .question_answer import QuestionAnswerDataModule

__ALL__ = [
    "DocClassificationDataModule",
    "QuestionAnswerDataModule",
    "TextClassificationDataModule",
    "TranslationDataModule",
]