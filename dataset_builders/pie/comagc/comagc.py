import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import datasets
from pytorch_ie import AnnotationLayer, Document, annotation_field
from pytorch_ie.annotations import BinaryRelation, LabeledSpan, Span, MultiLabeledSpan
from pytorch_ie.documents import (
    TextBasedDocument,
    TextDocumentWithLabeledSpansAndBinaryRelations,
)

from pie_datasets import ArrowBasedBuilder

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NamedSpan(Span):
    name: str

    def resolve(self) -> Any:
        return self.name, super().resolve()


@dataclass(frozen=True)
class SpanWithNameAndType(Span):
    name: str
    type: str

    def resolve(self) -> Any:
        return self.name, self.type, super().resolve()


@dataclass
class ComagcDocument(Document):
    pmid: str
    sentence: str
    cge: str
    ccs: str
    cancer_type: str
    gene: AnnotationLayer[NamedSpan] = annotation_field(target="sentence")
    cancer: AnnotationLayer[NamedSpan] = annotation_field(target="sentence")
    pt: Optional[str] = None
    ige: Optional[str] = None
    expression_change_keyword1: AnnotationLayer[SpanWithNameAndType] = annotation_field(target="sentence")
    expression_change_keyword2: AnnotationLayer[SpanWithNameAndType] = annotation_field(target="sentence")


def example_to_document(example) -> ComagcDocument:

    doc = ComagcDocument(
        pmid=example["pmid"],
        sentence=example["sentence"],
        cancer_type=example["cancer_type"],
        cge=example["CGE"],
        ccs=example["CCS"],
        pt=example["PT"],
        ige=example["IGE"],
    )

    # Gene and cancer entities
    # name is (almost) always the text of the gene/cancer (between the start and end position)
    gene = NamedSpan(
        start=example["gene"]["pos"][0],
        end=example["gene"]["pos"][1] + 1,
        name=example["gene"]["name"],
    )
    doc.gene.extend([gene])

    cancer = NamedSpan(
        start=example["cancer"]["pos"][0],
        end=example["cancer"]["pos"][1] + 1,
        name=example["cancer"]["name"],
    )
    doc.cancer.extend([cancer])

    # Expression change keywords
    # expression_change_keyword_1 might have no values
    if example["expression_change_keyword_1"]["pos"] is not None:
        expression_change_keyword1 = SpanWithNameAndType(
            start=example["expression_change_keyword_1"]["pos"][0],
            end=example["expression_change_keyword_1"]["pos"][1] + 1,
            name=example["expression_change_keyword_1"]["name"],
            type=example["expression_change_keyword_1"]["type"],
        )
        doc.expression_change_keyword1.extend([expression_change_keyword1])

    expression_change_keyword2 = SpanWithNameAndType(
        start=example["expression_change_keyword_2"]["pos"][0],
        end=example["expression_change_keyword_2"]["pos"][1] + 1,
        name=example["expression_change_keyword_2"]["name"],
        type=example["expression_change_keyword_2"]["type"],
    )
    doc.expression_change_keyword2.extend([expression_change_keyword2])

    return doc


def document_to_example(doc: ComagcDocument) -> Dict[str, Any]:

    gene = {
        "name": doc.gene[0].name,
        "pos": [doc.gene[0].start, doc.gene[0].end - 1],
    }
    cancer = {
        "name": doc.cancer[0].name,
        "pos": [doc.cancer[0].start, doc.cancer[0].end - 1],
    }

    if doc.expression_change_keyword1[0].end == 0:
        pos = None
    else:
        pos = [doc.expression_change_keyword1[0].start, doc.expression_change_keyword1[0].end - 1]

    expression_change_keyword_1 = {
        "name": doc.expression_change_keyword1[0].name,
        "pos": pos,
        "type": doc.expression_change_keyword1[0].type,
    }
    expression_change_keyword_2 = {
        "name": doc.expression_change_keyword2[0].label[0],
        "pos": [doc.expression_change_keyword2[0].start, doc.expression_change_keyword2[0].end - 1],
        "type": doc.expression_change_keyword2[0].label[1],
    }

    return {
        "pmid": doc.id,
        "sentence": doc.text,
        "cancer_type": doc.metadata["cancer_type"],
        "gene": gene,
        "cancer": cancer,
        "CGE": doc.cge,
        "CCS": doc.ccs,
        "PT": doc.pt,
        "IGE": doc.ige,
        "expression_change_keyword_1": expression_change_keyword_1,
        "expression_change_keyword_2": expression_change_keyword_2,
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
