from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import datasets
from pytorch_ie import Document
from pytorch_ie.annotations import BinaryRelation, LabeledSpan
from pytorch_ie.documents import AnnotationLayer, TextBasedDocument, annotation_field

from pie_datasets import GeneratorBasedBuilder


@dataclass
class ChemprotDocument(TextBasedDocument):
    # check if correct
    title: Optional[str] = None
    abstract: Optional[str] = None
    entities: AnnotationLayer[LabeledSpan] = annotation_field(target="text")
    relations: AnnotationLayer[BinaryRelation] = annotation_field(target="entities")


@dataclass
class ChemprotBigbioDocument(TextBasedDocument):
    # check if correct
    passages: AnnotationLayer[LabeledSpan] = annotation_field(target="text")
    entities: AnnotationLayer[LabeledSpan] = annotation_field(target="text")
    relations: AnnotationLayer[BinaryRelation] = annotation_field(target="entities")


@dataclass
class ChemprotTaskEvalDocument(TextBasedDocument):
    # TODO
    pass


class ChemprotConfig(datasets.BuilderConfig):
    pass


class Chemprot(GeneratorBasedBuilder):

    DOCUMENT_TYPES = {
        "chemprot_full_source": ChemprotDocument,
        "chemprot_bigbio_kb": ChemprotBigbioDocument,
        "chemprot_shared_task_eval_source": ChemprotTaskEvalDocument,
    }

    BASE_DATASET_PATH = "bigbio/chemprot"
    BASE_DATASET_REVISION = "xyz"  # TODO

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
        pass

    def _generate_example(self, document: Document, **kwargs) -> Dict[str, Any]:
        pass
