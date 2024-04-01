from dataclasses import dataclass
from typing import Any, Dict, Optional

import datasets
from pytorch_ie.annotations import BinaryRelation, Label, LabeledSpan
from pytorch_ie.core import AnnotationLayer, annotation_field
from pytorch_ie.documents import (
    TextBasedDocument,
    TextDocumentWithLabeledSpansAndBinaryRelations,
)

from pie_datasets import GeneratorBasedBuilder


@dataclass
class DrugprotDocument(TextBasedDocument):
    title: Optional[str] = None
    abstract: Optional[str] = None
    entities: AnnotationLayer[LabeledSpan] = annotation_field(target="text")
    relations: AnnotationLayer[BinaryRelation] = annotation_field(target="entities")


@dataclass
class DrugprotBigbioDocument(TextBasedDocument):
    passages: AnnotationLayer[LabeledSpan] = annotation_field(target="text")
    entities: AnnotationLayer[LabeledSpan] = annotation_field(target="text")
    relations: AnnotationLayer[BinaryRelation] = annotation_field(target="entities")


def example2drugprot(example: Dict[str, Any]) -> DrugprotDocument:
    metadata = {"entity_id": []}
    id2start = {}
    for entity in example["entities"]:
        id2start[entity["id"]] = entity["offset"][0]

    document = DrugprotDocument(
        text=example["text"],
        title=example["title"],
        abstract=example["abstract"],
        id=example["document_id"],
        metadata=metadata,
    )
    # We sort labels and relation to always have an deterministic order for testing purposes.
    for span in sorted(example["entities"], key=lambda span: span["offset"][0]):
        labeled_span = LabeledSpan(
            start=span["offset"][0],
            end=span["offset"][1],
            label=span["type"],
        )
        document.entities.append(labeled_span)
        document.metadata["entity_id"].append(span["id"])
    for relation in sorted(example["relations"], key=lambda relation: relation["id"]):
        document.relations.append(
            BinaryRelation(
                head=[
                    span
                    for span in document.entities
                    if span.start == id2start[relation["arg1_id"]]
                ][0],
                tail=[
                    span
                    for span in document.entities
                    if span.start == id2start[relation["arg2_id"]]
                ][0],
                label=relation["type"],
            )
        )
    return document


def example2drugprot_bigbio(example: Dict[str, Any]) -> DrugprotBigbioDocument:
    text = " ".join([" ".join(passage["text"]) for passage in example["passages"]])
    doc_id = example["document_id"]
    metadata = {"entity_id": []}

    id2start = {}
    for entity in example["entities"]:
        id2start[entity["id"]] = entity["offsets"][0][0]

    document = DrugprotBigbioDocument(
        text=text,
        id=doc_id,
        metadata=metadata,
    )
    for passage in example["passages"]:
        document.passages.append(
            LabeledSpan(
                start=passage["offsets"][0][0],
                end=passage["offsets"][0][1],
                label=passage["type"],
            )
        )
    # We sort labels and relation to always have an deterministic order for testing purposes.
    for span in sorted(example["entities"], key=lambda span: span["offsets"][0][0]):
        document.entities.append(
            LabeledSpan(
                start=span["offsets"][0][0],
                end=span["offsets"][0][1],
                label=span["type"],
            )
        )
        document.metadata["entity_id"].append(span["id"])
    for relation in sorted(example["relations"], key=lambda relation: relation["id"]):
        document.relations.append(
            BinaryRelation(
                head=[
                    span
                    for span in document.entities
                    if span.start == id2start[relation["arg1_id"]]
                ][0],
                tail=[
                    span
                    for span in document.entities
                    if span.start == id2start[relation["arg2_id"]]
                ][0],
                label=relation["type"],
            )
        )
    return document


class Drugprot(GeneratorBasedBuilder):
    DOCUMENT_TYPES = {
        "drugprot_source": DrugprotDocument,
        "drugprot_bigbio_kb": DrugprotBigbioDocument,
    }

    BASE_DATASET_PATH = "bigbio/drugprot"
    BASE_DATASET_REVISION = "38ff03d68347aaf694e598c50cb164191f50f61c"

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="drugprot_source",
            version=datasets.Version("1.0.2"),
            description="DrugProt source schema",
        ),
        datasets.BuilderConfig(
            name="drugprot_bigbio_kb",
            version=datasets.Version("1.0.0"),
            description="DrugProt BigBio schema",
        ),
    ]
    DOCUMENT_CONVERTERS = {
        TextDocumentWithLabeledSpansAndBinaryRelations: {
            "entities": "labeled_spans",
            "relations": "binary_relations",
        }
    }
    # @property
    # def document_converters(self):
    #     if self.config.name in ("drugprot_source", "drugprot_bigbio_kb"):
    #         return {
    #             TextDocumentWithLabeledSpansAndBinaryRelations: {
    #                 "entities": "labeled_spans",
    #                 "relations": "binary_relations",
    #             }
    #         }
    #     else:
    #         raise ValueError(f"Unknown dataset name: {self.config.name}")

    def _generate_document(
        self,
        example: Dict[str, Any],
    ) -> DrugprotDocument | DrugprotBigbioDocument:
        if self.config.name == "drugprot_source":
            return example2drugprot(example)
        elif self.config.name == "drugprot_bigbio_kb":
            return example2drugprot_bigbio(example)
