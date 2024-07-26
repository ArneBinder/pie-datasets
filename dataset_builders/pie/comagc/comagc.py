import logging
from dataclasses import dataclass
from typing import Any, Dict

import datasets
from pytorch_ie import AnnotationLayer, Document, annotation_field
from pytorch_ie.annotations import BinaryRelation, LabeledSpan, Span, MultiLabeledSpan
from pytorch_ie.documents import (
    TextBasedDocument,
    TextDocumentWithLabeledSpansAndBinaryRelations,
)

from pie_datasets import ArrowBasedBuilder

logger = logging.getLogger(__name__)


@dataclass
class ComagcDocument(TextBasedDocument):
    cancer_type: str = None
    gene: AnnotationLayer[LabeledSpan] = annotation_field(target="text")
    cancer: AnnotationLayer[LabeledSpan] = annotation_field(target="text")
    cge: str = None
    ccs: str = None
    pt: str = None
    ige: str = None
    expression_change_keywords: AnnotationLayer[MultiLabeledSpan] = annotation_field(target="text")


def example_to_document(example) -> ComagcDocument:
    metadata = {
        "cancer_type": example["cancer_type"],
        "annotation": {
            "CGE": example["CGE"],
            "CCS": example["CCS"],
            "PT": example["PT"],
            "IGE": example["IGE"],
        },
        "expression_change_keywords": [
            example["expression_change_keyword_1"],
            example["expression_change_keyword_2"],
        ],
    }

    doc = ComagcDocument(
        text=example["sentence"],
        id=example["pmid"],
        metadata=metadata,
        cancer_type=example["cancer_type"],
        cge=example["CGE"],
        ccs=example["CCS"],
        pt=example["PT"],
        ige=example["IGE"],
    )

    # name is (almost) always the text of the gene/cancer (between the start and end position)
    gene = LabeledSpan(
        start=example["gene"]["pos"][0],
        end=example["gene"]["pos"][1] + 1,
        label=example["gene"]["name"],
    )
    doc.gene.extend([gene])

    cancer = LabeledSpan(
        start=example["cancer"]["pos"][0],
        end=example["cancer"]["pos"][1] + 1,
        label=example["cancer"]["name"],
    )
    doc.cancer.extend([cancer])

    # expression_change_keyword can have None for ["pos"]
    """expression_change_keyword1 = MultiLabeledSpan(
        start=example["expression_change_keyword_1"]["pos"][0],
        end=example["expression_change_keyword_1"]["pos"][1] + 1,
        label=(example["expression_change_keyword_1"]["name"], example["expression_change_keyword_1"]["type"]),
    )
    expression_change_keyword2 = MultiLabeledSpan(
        start=example["expression_change_keyword_2"]["pos"][0],
        end=example["expression_change_keyword_2"]["pos"][1] + 1,
        label=(example["expression_change_keyword_2"]["name"], example["expression_change_keyword_2"]["type"]),
    )
    doc.expression_change_keywords.extend([expression_change_keyword1, expression_change_keyword2])"""

    return doc


def document_to_example(doc: ComagcDocument) -> Dict[str, Any]:
    # still need to adjust
    gene = {
        "name": doc.entities[0].label,
        "pos": [doc.entities[0].start, doc.entities[0].end - 1],
    }
    cancer = {
        "name": doc.entities[1].label,
        "pos": [doc.entities[1].start, doc.entities[1].end - 1],
    }

    return {
        "pmid": doc.id,
        "sentence": doc.text,
        "cancer_type": doc.metadata["cancer_type"],
        "gene": gene,
        "cancer": cancer,
        "CGE": doc.metadata["annotation"]["CGE"],
        "CCS": doc.metadata["annotation"]["CCS"],
        "PT": doc.metadata["annotation"]["PT"],
        "IGE": doc.metadata["annotation"]["IGE"],
        "expression_change_keyword_1": doc.metadata["expression_change_keywords"][0],
        "expression_change_keyword_2": doc.metadata["expression_change_keywords"][1],
    }


class Comagc(ArrowBasedBuilder):
    DOCUMENT_TYPE = ComagcDocument
    BASE_DATASET_PATH = "DFKI-SLT/CoMAGC"
    BASE_DATASET_REVISION = "8e2950b8a3967c2f45de86f60dd5c8ccb9ad3815"

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            version=datasets.Version("1.0.0"),
            description="CoMAGC dataset",
        )
    ]

    @property
    def document_converters(self):
        return {
            TextDocumentWithLabeledSpansAndBinaryRelations: {
                "entities": "labeled_spans",
                "relations": "binary_relations",
            }
        }

    def _generate_document(self, example, **kwargs):
        return example_to_document(example)

    def _generate_example(self, document: Document, **kwargs) -> Dict[str, Any]:
        return document_to_example(document)


def get_relation_label(example: Dict) -> str:
    """Simple rule-based function to determine the relation between the gene and the cancer.

    As this dataset contains a multi-faceted annotation scheme
    for gene-cancer relations, it does not only label the relation
    between gene and cancer, but provides further information.
    However, the relation of interest stays the gene-class,
    which can be derived from inference rules
    (https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-14-323/tables/3), based on the
    information given in columns CGE, CCS, IGE, PT.
    """

    rules = [
        {
            "CGE": "increased",
            "CCS": "normalTOcancer",
            "IGE": "*",
            "PT": "causality",
            "Gene class": "oncogene",
        },
        {
            "CGE": "decreased",
            "CCS": "cancerTOnormal",
            "IGE": "unidentifiable",
            "PT": "causality",
            "Gene class": "oncogene",
        },
        {
            "CGE": "decreased",
            "CCS": "cancerTOnormal",
            "IGE": "up-regulated",
            "PT": "*",
            "Gene class": "oncogene",
        },
        {
            "CGE": "decreased",
            "CCS": "normalTOcancer",
            "IGE": "*",
            "PT": "causality",
            "Gene class": "tumor suppressor gene",
        },
        {
            "CGE": "increased",
            "CCS": "cancerTOnormal",
            "IGE": "unidentifiable",
            "PT": "causality",
            "Gene class": "tumor suppressor gene",
        },
        {
            "CGE": "increased",
            "CCS": "cancerTOnormal",
            "IGE": "down-regulated",
            "PT": "*",
            "Gene class": "tumor suppressor gene",
        },
        {
            "CGE": "*",
            "CCS": "normalTOcancer",
            "IGE": "*",
            "PT": "observation",
            "Gene class": "biomarker",
        },
        {
            "CGE": "*",
            "CCS": "cancerTOnormal",
            "IGE": "unidentifiable",
            "PT": "observation",
            "Gene class": "biomarker",
        },
        {
            "CGE": "decreased",
            "CCS": "cancer->cancer",
            "IGE": "up-regulated",
            "PT": "observation",
            "Gene class": "biomarker",
        },
        {
            "CGE": "increased",
            "CCS": "cancer->cancer",
            "IGE": "down-regulated",
            "PT": "observation",
            "Gene class": "biomarker",
        },
    ]

    for rule in rules:
        if (
            (rule["CGE"] == "*" or example["CGE"] == rule["CGE"])
            and (rule["CCS"] == "*" or example["CCS"] == rule["CCS"])
            and (rule["IGE"] == "*" or example["IGE"] == rule["IGE"])
            and (rule["PT"] == "*" or example["PT"] == rule["PT"])
        ):
            return rule["Gene class"]

    # logger.warning("No rule matched.") # turned off to avoid spamming the logs
    # NOTE: The label "UNIDENTIFIED" is not part of the original dataset, but added for the sake of completeness
    return "UNIDENTIFIED"
