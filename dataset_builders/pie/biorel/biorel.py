import dataclasses
from typing import Any

import datasets
from pytorch_ie import AnnotationLayer, annotation_field
from pytorch_ie.annotations import BinaryRelation, LabeledSpan, Span
from pytorch_ie.documents import (
    TextBasedDocument,
    TextDocumentWithLabeledSpansAndBinaryRelations,
)

from pie_datasets import ArrowBasedBuilder, GeneratorBasedBuilder


@dataclasses.dataclass(frozen=True)
class SpanWithIdAndName(Span):
    id: str
    name: str

    def resolve(self) -> Any:
        return self.id, self.name, super().resolve()


@dataclasses.dataclass
class BioRelDocument(TextBasedDocument):
    entities: AnnotationLayer[SpanWithIdAndName] = annotation_field(target="text")
    relations: AnnotationLayer[BinaryRelation] = annotation_field(target="entities")


def example_to_document(example):
    document = BioRelDocument(text=example["text"])
    head = SpanWithIdAndName(
        id=example["h"]["id"],
        name=example["h"]["name"],
        start=example["h"]["pos"][0],
        end=example["h"]["pos"][1],
    )
    tail = SpanWithIdAndName(
        id=example["t"]["id"],
        name=example["t"]["name"],
        start=example["t"]["pos"][0],
        end=example["t"]["pos"][1],
    )
    document.entities.extend([head, tail])

    relation = BinaryRelation(head=head, tail=tail, label=example["relation"])
    document.relations.append(relation)
    return document


def document_to_example(document):
    head = document.entities[0]
    tail = document.entities[1]
    return {
        "text": document.text,
        "relation": document.relations[0].label,
        "h": {"id": head.id, "name": head.name, "pos": [head.start, head.end]},
        "t": {"id": tail.id, "name": tail.name, "pos": [tail.start, tail.end]},
    }


class BioRelConfig(datasets.BuilderConfig):
    """BuilderConfig for BioRel."""

    pass


class BioRel(ArrowBasedBuilder):
    # set the correct values for the following attributes
    DOCUMENT_TYPE = BioRelDocument
    BASE_DATASET_PATH = "DFKI-SLT/BioRel"
    BASE_DATASET_REVISION = "e4869c484c582cfbc7ead10d4d421bd4b275fa4e"
    # BASE_CONFIG_KWARGS_DICT = None

    BUILDER_CONFIGS = [
        BioRelConfig(
            name="biorel",
            version=datasets.Version("1.0.0"),
            description="BioRel dataset",
        )
    ]

    DOCUMENT_CONVERTERS = {
        TextDocumentWithLabeledSpansAndBinaryRelations: {
            "entities": "labeled_spans",
            "relations": "binary_relations",
        }
    }

    def _generate_document(self, example, **kwargs):
        return example_to_document(example)
