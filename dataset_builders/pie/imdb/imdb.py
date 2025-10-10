from dataclasses import dataclass
from typing import Any, Dict

import datasets
from pie_documents.annotations import Label
from pie_documents.documents import TextDocumentWithLabel

from pie_datasets import GeneratorBasedBuilder


@dataclass
class ImdbDocument(TextDocumentWithLabel):
    pass


def example_to_document(example: Dict[str, Any], labels: datasets.ClassLabel) -> ImdbDocument:
    text = example["text"]
    document = ImdbDocument(text=text)
    label_id = example["label"]
    if label_id < 0:
        return document

    label = labels.int2str(label_id)
    label_annotation = Label(label=label)
    document.label.append(label_annotation)

    return document


def document_to_example(document: ImdbDocument, labels: datasets.ClassLabel) -> Dict[str, Any]:
    if len(document.label) > 0:
        label_id = labels.str2int(document.label[0].label)
    else:
        label_id = -1

    return {
        "text": document.text,
        "label": label_id,
    }


class Imdb(GeneratorBasedBuilder):
    DOCUMENT_TYPE = ImdbDocument

    BASE_DATASET_PATH = "stanfordnlp/imdb"
    BASE_DATASET_REVISION = "9c6ede893febf99215a29cc7b72992bb1138b06b"

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="plain_text",
            version=datasets.Version("1.0.0"),
            description="IMDB sentiment classification dataset",
        ),
    ]

    DOCUMENT_CONVERTERS = {TextDocumentWithLabel: {}}

    def _generate_document_kwargs(self, dataset) -> Dict[str, Any]:
        return {"labels": dataset.features["label"]}

    def _generate_document(self, example, **kwargs) -> ImdbDocument:
        return example_to_document(example, **kwargs)

    def _generate_example_kwargs(self, dataset) -> Dict[str, Any]:
        return {"labels": dataset.features["label"]}

    def _generate_example(self, document: ImdbDocument, **kwargs) -> Dict[str, Any]:
        return document_to_example(document, **kwargs)
