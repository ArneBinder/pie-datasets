from dataclasses import dataclass
from typing import Any, Dict

import datasets
from pie_core import Document
from pie_modules.annotations import BinaryRelation, LabeledSpan
from pie_modules.documents import (
    AnnotationLayer,
    TextBasedDocument,
    TextDocumentWithLabeledSpansAndBinaryRelations,
    TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions,
    annotation_field,
)

from pie_datasets import GeneratorBasedBuilder


@dataclass
class ChemprotDocument(TextBasedDocument):
    # used by chemprot_full_source and chemprot_shared_task_eval_source
    entities: AnnotationLayer[LabeledSpan] = annotation_field(target="text")
    relations: AnnotationLayer[BinaryRelation] = annotation_field(target="entities")


@dataclass
class ChemprotBigbioDocument(TextBasedDocument):
    passages: AnnotationLayer[LabeledSpan] = annotation_field(target="text")
    entities: AnnotationLayer[LabeledSpan] = annotation_field(target="text")
    relations: AnnotationLayer[BinaryRelation] = annotation_field(target="entities")


def example_to_chemprot_doc(example) -> ChemprotDocument:
    metadata = {"entity_ids": []}
    id_to_labeled_span: Dict[str, LabeledSpan] = {}

    doc = ChemprotDocument(
        text=example["text"],
        id=example["pmid"],
        metadata=metadata,
    )

    for idx in range(len(example["entities"]["id"])):
        labeled_span = LabeledSpan(
            start=example["entities"]["offsets"][idx][0],
            end=example["entities"]["offsets"][idx][1],
            label=example["entities"]["type"][idx],
        )
        doc.entities.append(labeled_span)
        doc.metadata["entity_ids"].append(example["entities"]["id"][idx])
        id_to_labeled_span[example["entities"]["id"][idx]] = labeled_span

    for idx in range(len(example["relations"]["type"])):
        doc.relations.append(
            BinaryRelation(
                head=id_to_labeled_span[example["relations"]["arg1"][idx]],
                tail=id_to_labeled_span[example["relations"]["arg2"][idx]],
                label=example["relations"]["type"][idx],
            )
        )

    return doc


def example_to_chemprot_bigbio_doc(example) -> ChemprotBigbioDocument:
    text = " ".join([" ".join(passage["text"]) for passage in example["passages"]])
    metadata = {"id": example["id"], "entity_ids": [], "relation_ids": []}
    id_to_labeled_span: Dict[str, LabeledSpan] = {}

    doc = ChemprotBigbioDocument(
        text=text,
        id=example["document_id"],
        metadata=metadata,
    )

    for passage in example["passages"]:
        doc.passages.append(
            LabeledSpan(
                start=passage["offsets"][0][0],
                end=passage["offsets"][0][1],
                label=passage["type"],
            )
        )

    for span in example["entities"]:
        labeled_span = LabeledSpan(
            start=span["offsets"][0][0],
            end=span["offsets"][0][1],
            label=span["type"],
        )
        doc.entities.append(labeled_span)
        doc.metadata["entity_ids"].append(span["id"])
        id_to_labeled_span[span["id"]] = labeled_span

    for relation in example["relations"]:
        doc.relations.append(
            BinaryRelation(
                head=id_to_labeled_span[relation["arg1_id"]],
                tail=id_to_labeled_span[relation["arg2_id"]],
                label=relation["type"],
            )
        )
        doc.metadata["relation_ids"].append([relation["arg1_id"], relation["arg2_id"]])

    return doc


def chemprot_doc_to_example(doc: ChemprotDocument) -> Dict[str, Any]:
    entities = {
        "id": [],
        "offsets": [],
        "text": [],
        "type": [],
    }
    relations = {
        "arg1": [],
        "arg2": [],
        "type": [],
    }

    entity_id2entity = {
        ent_id: entity for ent_id, entity in zip(doc.metadata["entity_ids"], doc.entities)
    }

    for entity_id, entity in zip(doc.metadata["entity_ids"], doc.entities):
        entities["id"].append(entity_id)
        entities["offsets"].append([entity.start, entity.end])
        entities["text"].append(doc.text[entity.start : entity.end])
        entities["type"].append(entity.label)

        if entity in entity_id2entity:
            raise ValueError("Entity already exists in entity_id2entity")

        entity_id2entity[entity] = entity_id

    for relation in doc.relations:
        relations["arg1"].append(entity_id2entity[relation.head])
        relations["arg2"].append(entity_id2entity[relation.tail])
        relations["type"].append(relation.label)

    return {
        "text": doc.text,
        "pmid": doc.id,
        "entities": entities,
        "relations": relations,
    }


def chemprot_bigbio_doc_to_example(doc: ChemprotBigbioDocument) -> Dict[str, Any]:
    id = int(doc.metadata["id"])
    passages = []
    entities = []
    relations = []

    entity_id2entity = {
        ent_id: entity for ent_id, entity in zip(doc.metadata["entity_ids"], doc.entities)
    }

    for passage in doc.passages:
        id += 1
        passages.append(
            {
                "id": str(id),
                "offsets": [[passage.start, passage.end]],
                "text": [doc.text[passage.start : passage.end]],
                "type": passage.label,
            }
        )

    entity2entity_id = dict()

    for entity_id, entity in zip(doc.metadata["entity_ids"], doc.entities):
        id += 1
        entities.append(
            {
                "id": entity_id,  # entity_id = str(id)
                "normalized": [],
                "offsets": [[entity.start, entity.end]],
                "text": [doc.text[entity.start : entity.end]],
                "type": entity.label,
            }
        )
        if entity in entity_id2entity:
            raise ValueError("Entity already exists in entity_id2entity")

        entity2entity_id[entity] = entity_id

    for relation in doc.relations:
        id += 1
        relations.append(
            {
                "id": str(id),  # save in metadata?
                "arg1_id": entity2entity_id[relation.head],
                "arg2_id": entity2entity_id[relation.tail],
                "type": relation.label,
                "normalized": [],
            }
        )

    return {
        "id": doc.metadata["id"],
        "document_id": doc.id,
        "passages": passages,
        "entities": entities,
        "events": [],
        "coreferences": [],
        "relations": relations,
    }


class Chemprot(GeneratorBasedBuilder):
    DOCUMENT_TYPES = {  # Note ChemprotDocument is used twice
        "chemprot_full_source": ChemprotDocument,
        "chemprot_bigbio_kb": ChemprotBigbioDocument,
        "chemprot_shared_task_eval_source": ChemprotDocument,
    }

    BASE_DATASET_PATH = "DFKI-SLT/chemprot"
    BASE_DATASET_REVISION = "0ac36434e431c5e74c40ccf1baa0fdac6f7698ee"

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="chemprot_full_source",
            version=datasets.Version("1.0.0"),
            description="ChemProt full source version",
        ),
        datasets.BuilderConfig(
            name="chemprot_bigbio_kb",
            version=datasets.Version("1.0.0"),
            description="ChemProt BigBio kb version",
        ),
        datasets.BuilderConfig(
            name="chemprot_shared_task_eval_source",
            version=datasets.Version("1.0.0"),
            description="ChemProt shared task eval source version",
        ),
    ]

    @property
    def document_converters(self):
        if (
            self.config.name == "chemprot_full_source"
            or self.config.name == "chemprot_shared_task_eval_source"
        ):
            return {
                TextDocumentWithLabeledSpansAndBinaryRelations: {
                    "entities": "labeled_spans",
                    "relations": "binary_relations",
                }
            }
        elif self.config.name == "chemprot_bigbio_kb":
            return {
                TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions: {
                    "passages": "labeled_partitions",
                    "entities": "labeled_spans",
                    "relations": "binary_relations",
                }
            }
        else:
            raise ValueError(f"Unknown dataset name: {self.config.name}")

    def _generate_document(self, example, **kwargs):
        if self.config.name == "chemprot_bigbio_kb":
            return example_to_chemprot_bigbio_doc(example)
        else:
            return example_to_chemprot_doc(example)

    def _generate_example(self, document: Document, **kwargs) -> Dict[str, Any]:
        if isinstance(document, ChemprotBigbioDocument):
            return chemprot_bigbio_doc_to_example(document)
        elif isinstance(document, ChemprotDocument):
            return chemprot_doc_to_example(document)
        else:
            raise ValueError(f"Unknown document type: {type(document)}")
