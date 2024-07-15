import dataclasses
import logging
from typing import Any

import datasets
from pytorch_ie import AnnotationLayer, annotation_field
from pytorch_ie.annotations import BinaryRelation, LabeledSpan, Span
from pytorch_ie.documents import (
    TextBasedDocument,
    TextDocumentWithLabeledSpansAndBinaryRelations,
)

from pie_datasets import ArrowBasedBuilder, GeneratorBasedBuilder

logger = logging.getLogger(__name__)


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


def convert_to_text_document_with_labeled_spans_and_binary_relations(
    document: BioRelDocument,
) -> TextDocumentWithLabeledSpansAndBinaryRelations:
    text_document = TextDocumentWithLabeledSpansAndBinaryRelations(text=document.text)
    old2new_spans = {}
    for entity in document.entities:  # in our case two entities (head and tail)
        # create LabeledSpan and append (required for document type)
        labeled_span = LabeledSpan(start=entity.start, end=entity.end, label="ENTITY")
        text_document.labeled_spans.append(labeled_span)

        # check if the labeled span text is the same as the entity name
        if str(labeled_span) != entity.name:
            logger.warning(
                f"Expected labeled span text to be '{entity.name}', got '{labeled_span}'"
            )

        # Map the original entity to the new labeled span
        old2new_spans[entity] = labeled_span

    if len(document.relations) != 1:  # one relation between two entities
        raise ValueError(f"Expected exactly one relation, got {len(document.relations)}")
    old_rel = document.relations[0]

    # create BinaryRelation and append (required for document type)
    rel = BinaryRelation(
        head=old2new_spans[old_rel.head],
        tail=old2new_spans[old_rel.tail],
        label=old_rel.label,
    )
    text_document.binary_relations.append(rel)
    return text_document


class BioRelConfig(datasets.BuilderConfig):
    """BuilderConfig for BioRel."""

    pass


class BioRel(ArrowBasedBuilder):
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
        TextDocumentWithLabeledSpansAndBinaryRelations: convert_to_text_document_with_labeled_spans_and_binary_relations
    }

    def _generate_document(self, example, **kwargs):
        return example_to_document(example)

    def _generate_example(self, document: BioRelDocument, **kwargs):
        return document_to_example(document)
