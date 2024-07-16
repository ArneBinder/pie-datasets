from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import datasets
from pytorch_ie import Document
from pytorch_ie.annotations import BinaryRelation, LabeledSpan
from pytorch_ie.documents import AnnotationLayer, TextBasedDocument, annotation_field

from pie_datasets import GeneratorBasedBuilder


@dataclass
class ChemprotDocument(TextBasedDocument):
    # used by chemprot_full_source and chemprot_shared_task_eval_source
    entities: AnnotationLayer[LabeledSpan] = annotation_field(target="text")
    relations: AnnotationLayer[BinaryRelation] = annotation_field(target="entities")


@dataclass
class ChemprotBigbioDocument(TextBasedDocument):
    # check if correct
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
        # entities have "text" field: already included through the offset?
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
    metadata = {"entity_ids": []}
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
        # entities have "text" field: already included through the offset?
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

    return doc


def chemprot_doc_to_example(doc: ChemprotDocument) -> Dict[str, Any]:
    # still in the process of being implemented
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

    entities["id"] = doc.metadata["entity_ids"]
    for entity in doc.entities:
        entities["offsets"].append([entity.start, entity.end])
        entities["text"].append(doc.text[entity.start : entity.end])
        entities["type"].append(entity.label)

    for relation in doc.relations:
        # relations["arg1"].append(relation.head.id)
        # relations["arg2"].append(relation.tail.id)
        relations["type"].append(relation.label)

    return {
        "text": doc.text,
        "pmid": doc.id,
        "entities": entities,
        "relations": relations,
    }


class ChemprotConfig(datasets.BuilderConfig):
    pass


class Chemprot(GeneratorBasedBuilder):
    DOCUMENT_TYPES = {  # Note ChemprotDocument is used twice
        "chemprot_full_source": ChemprotDocument,
        "chemprot_bigbio_kb": ChemprotBigbioDocument,
        "chemprot_shared_task_eval_source": ChemprotDocument,
    }

    BASE_DATASET_PATH = "bigbio/chemprot"
    BASE_DATASET_REVISION = "86afccf3ccc614f817a7fad0692bf62fbc5ce469"

    BUILDER_CONFIGS = [
        ChemprotConfig(
            name="chemprot_full_source",
            version=datasets.Version("1.0.0"),
            description="ChemProt full source version",
        ),
        ChemprotConfig(
            name="chemprot_bigbio_kb",
            version=datasets.Version("1.0.0"),
            description="ChemProt BigBio kb version",
        ),
        ChemprotConfig(
            name="chemprot_shared_task_eval_source",
            version=datasets.Version("1.0.0"),
            description="ChemProt shared task eval source version",
        ),
    ]

    def _generate_document(self, example, **kwargs):
        if self.config.name == "chemprot_bigbio_kb":
            return example_to_chemprot_bigbio_doc(example)
        else:
            return example_to_chemprot_doc(example)

    def _generate_example(self, document: Document, **kwargs) -> Dict[str, Any]:
        pass
