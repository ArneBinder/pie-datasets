from dataclasses import dataclass

import datasets
from pytorch_ie.annotations import Label
from pytorch_ie.core import AnnotationList, annotation_field
from pytorch_ie.documents import TextBasedDocument

from pie_datasets import GeneratorBasedBuilder


class ImdbConfig(datasets.BuilderConfig):
    """BuilderConfig for IMDB."""

    def __init__(self, **kwargs):
        """BuilderConfig for IMDB.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super().__init__(**kwargs)


@dataclass
class ImdbDocument(TextBasedDocument):
    label: AnnotationList[Label] = annotation_field()


class Imdb(GeneratorBasedBuilder):
    DOCUMENT_TYPE = ImdbDocument

    BASE_DATASET_PATH = "imdb"
    BASE_DATASET_REVISION = "9c6ede893febf99215a29cc7b72992bb1138b06b"

    BUILDER_CONFIGS = [
        ImdbConfig(
            name="plain_text",
            version=datasets.Version("1.0.0"),
            description="IMDB sentiment classification dataset",
        ),
    ]

    def _generate_document_kwargs(self, dataset):
        return {"labels": dataset.features["label"]}

    def _generate_document(self, example, labels: datasets.ClassLabel):
        text = example["text"]
        document = ImdbDocument(text=text)
        label_id = example["label"]
        if label_id < 0:
            return document

        label = labels.int2str(label_id)
        label_annotation = Label(label=label)
        document.label.append(label_annotation)

        return document
