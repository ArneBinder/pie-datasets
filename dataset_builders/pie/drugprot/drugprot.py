from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import datasets
from pytorch_ie.annotations import BinaryRelation, LabeledSpan
from pytorch_ie.documents import (
    AnnotationLayer,
    TextBasedDocument,
    TextDocumentWithLabeledSpansAndBinaryRelations,
    TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions,
    annotation_field,
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
    metadata = {"entity_ids": []}
    id2labeled_span: Dict[str, LabeledSpan] = {}

    document = DrugprotDocument(
        text=example["text"],
        title=example["title"],
        abstract=example["abstract"],
        id=example["document_id"],
        metadata=metadata,
    )
    for span in example["entities"]:
        labeled_span = LabeledSpan(
            start=span["offset"][0],
            end=span["offset"][1],
            label=span["type"],
        )
        document.entities.append(labeled_span)
        document.metadata["entity_ids"].append(span["id"])
        id2labeled_span[span["id"]] = labeled_span
    for relation in example["relations"]:
        document.relations.append(
            BinaryRelation(
                head=id2labeled_span[relation["arg1_id"]],
                tail=id2labeled_span[relation["arg2_id"]],
                label=relation["type"],
            )
        )
    return document


def example2drugprot_bigbio(example: Dict[str, Any]) -> DrugprotBigbioDocument:
    text = " ".join([" ".join(passage["text"]) for passage in example["passages"]])
    doc_id = example["document_id"]
    metadata = {"entity_ids": []}
    id2labeled_span: Dict[str, LabeledSpan] = {}

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
    for span in example["entities"]:
        labeled_span = LabeledSpan(
            start=span["offsets"][0][0],
            end=span["offsets"][0][1],
            label=span["type"],
        )
        document.entities.append(labeled_span)
        document.metadata["entity_ids"].append(span["id"])
        id2labeled_span[span["id"]] = labeled_span
    for relation in example["relations"]:
        document.relations.append(
            BinaryRelation(
                head=id2labeled_span[relation["arg1_id"]],
                tail=id2labeled_span[relation["arg2_id"]],
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
    # This revision includes the "test_background" split (see https://github.com/bigscience-workshop/biomedical/pull/928)
    BASE_DATASET_REVISION = "0cc98b3d292242e69adcfd2c3e5eea94baaca8ea"

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="drugprot_source",
            version=datasets.Version("1.0.2"),
            description="DrugProt source version",
        ),
        datasets.BuilderConfig(
            name="drugprot_bigbio_kb",
            version=datasets.Version("1.0.0"),
            description="DrugProt BigBio version",
        ),
    ]

    @property
    def document_converters(self):
        if self.config.name == "drugprot_source":
            return {
                TextDocumentWithLabeledSpansAndBinaryRelations: {
                    "entities": "labeled_spans",
                    "relations": "binary_relations",
                }
            }
        elif self.config.name == "drugprot_bigbio_kb":
            return {
                TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions: {
                    "passages": "labeled_partitions",
                    "entities": "labeled_spans",
                    "relations": "binary_relations",
                }
            }
        else:
            raise ValueError(f"Unknown dataset name: {self.config.name}")

    def _generate_document(
        self,
        example: Dict[str, Any],
    ) -> Union[DrugprotDocument, DrugprotBigbioDocument]:
        if self.config.name == "drugprot_source":
            return example2drugprot(example)
        elif self.config.name == "drugprot_bigbio_kb":
            return example2drugprot_bigbio(example)
        else:
            raise ValueError(f"Unknown dataset config name: {self.config.name}")
