import dataclasses
from typing import Any

import datasets
from pytorch_ie import AnnotationLayer, annotation_field
from pytorch_ie.annotations import BinaryRelation, Span
from pytorch_ie.documents import TextBasedDocument

from pie_datasets import ArrowBasedBuilder, GeneratorBasedBuilder


@dataclasses.dataclass(frozen=True)
class SpanWithIdAndName(Span):
    id: str
    name: str

    def resolve(self) -> Any:
        return self.id, self.name, super().resolve()


@dataclasses.dataclass
class TbgaDocument(TextBasedDocument):
    entities: AnnotationLayer[SpanWithIdAndName] = annotation_field(target="text")
    relations: AnnotationLayer[BinaryRelation] = annotation_field(target="entities")


def example_to_document(example) -> TbgaDocument:
    document = TbgaDocument(text=example["text"])
    head = SpanWithIdAndName(
        # this is due to the original dataset having an integer id but string is required
        id=str(example["h"]["id"]),
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
        "h": {"id": int(head.id), "name": head.name, "pos": [head.start, head.end]},
        "t": {"id": tail.id, "name": tail.name, "pos": [tail.start, tail.end]},
    }


class Tbga(ArrowBasedBuilder):
    DOCUMENT_TYPE = TbgaDocument
    BASE_DATASET_PATH = "DFKI-SLT/tbga"
    BASE_DATASET_REVISION = "78575b79aa1c6ff7712bfa0f0eb0e3d01d80e9bc"

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            version=datasets.Version("1.0.0"),
            description="TBGA dataset",
        )
    ]

    def _generate_document(self, example, **kwargs):
        return example_to_document(example)

    def _generate_example(self, document, **kwargs):
        return document_to_example(document)
