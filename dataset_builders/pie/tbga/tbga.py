import dataclasses
from typing import Any

import datasets
from pie_core import AnnotationLayer, annotation_field
from pie_documents.annotations import BinaryRelation, LabeledSpan, Span
from pie_documents.documents import (
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
        end=example["h"]["pos"][0] + example["h"]["pos"][1],  # end is start + length
    )
    tail = SpanWithIdAndName(
        id=example["t"]["id"],
        name=example["t"]["name"],
        start=example["t"]["pos"][0],
        end=example["t"]["pos"][0] + example["t"]["pos"][1],  # end is start + length
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
        "h": {"id": int(head.id), "name": head.name, "pos": [head.start, head.end - head.start]},
        "t": {"id": tail.id, "name": tail.name, "pos": [tail.start, tail.end - tail.start]},
    }


def convert_to_text_document_with_labeled_spans_and_binary_relations(
    document: TbgaDocument,
) -> TextDocumentWithLabeledSpansAndBinaryRelations:
    text_document = TextDocumentWithLabeledSpansAndBinaryRelations(text=document.text)
    old2new_spans = {}
    ids = []
    names = []

    for entity in document.entities:  # in our case two entities (head and tail)
        # create LabeledSpan and append
        labeled_span = LabeledSpan(start=entity.start, end=entity.end, label="ENTITY")
        text_document.labeled_spans.append(labeled_span)

        # Map the original entity to the new labeled span
        old2new_spans[entity] = labeled_span

        ids.append(entity.id)
        names.append(entity.name)

    if len(document.relations) != 1:  # one relation between two entities
        raise ValueError(f"Expected exactly one relation, got {len(document.relations)}")
    old_rel = document.relations[0]

    # create BinaryRelation and append
    rel = BinaryRelation(
        head=old2new_spans[old_rel.head],
        tail=old2new_spans[old_rel.tail],
        label=old_rel.label,
    )
    text_document.binary_relations.append(rel)
    text_document.metadata["entity_ids"] = ids
    text_document.metadata["entity_names"] = names

    return text_document


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

    BASE_BUILDER_KWARGS_DICT = {
        dataset_variant: {"trust_remote_code": True}
        for dataset_variant in [None] + [config.name for config in BUILDER_CONFIGS]
    }

    DOCUMENT_CONVERTERS = {
        TextDocumentWithLabeledSpansAndBinaryRelations: convert_to_text_document_with_labeled_spans_and_binary_relations
    }

    def _generate_document(self, example, **kwargs):
        return example_to_document(example)

    def _generate_example(self, document, **kwargs):
        return document_to_example(document)
