import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import datasets
from pytorch_ie import AnnotationLayer, Document, annotation_field
from pytorch_ie.annotations import BinaryRelation, LabeledSpan, Span
from pytorch_ie.documents import TextDocumentWithLabeledSpansAndBinaryRelations

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
    expression_change_keyword1: AnnotationLayer[SpanWithNameAndType] = annotation_field(
        target="sentence"
    )
    expression_change_keyword2: AnnotationLayer[SpanWithNameAndType] = annotation_field(
        target="sentence"
    )


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

    if not doc.expression_change_keyword1.resolve():
        expression_change_keyword_1 = {
            "name": "\nNone\n",
            "pos": None,
            "type": None,
        }
    else:
        expression_change_keyword_1 = {
            "name": doc.expression_change_keyword1[0].name,
            "pos": [
                doc.expression_change_keyword1[0].start,
                doc.expression_change_keyword1[0].end - 1,
            ],
            "type": doc.expression_change_keyword1[0].type,
        }

    expression_change_keyword_2 = {
        "name": doc.expression_change_keyword2[0].name,
        "pos": [
            doc.expression_change_keyword2[0].start,
            doc.expression_change_keyword2[0].end - 1,
        ],
        "type": doc.expression_change_keyword2[0].type,
    }

    return {
        "pmid": doc.pmid,
        "sentence": doc.sentence,
        "cancer_type": doc.cancer_type,
        "gene": gene,
        "cancer": cancer,
        "CGE": doc.cge,
        "CCS": doc.ccs,
        "PT": doc.pt,
        "IGE": doc.ige,
        "expression_change_keyword_1": expression_change_keyword_1,
        "expression_change_keyword_2": expression_change_keyword_2,
    }


def convert_to_text_document_with_labeled_spans_and_binary_relations(
    document: ComagcDocument,
) -> TextDocumentWithLabeledSpansAndBinaryRelations:
    metadata = {
        "cancer_type": document.cancer_type,
        "CGE": document.cge,
        "CCS": document.ccs,
        "PT": document.pt,
        "IGE": document.ige,
        "expression_change_keyword_1": document_to_example(document)[
            "expression_change_keyword_1"
        ],
        "expression_change_keyword_2": document_to_example(document)[
            "expression_change_keyword_2"
        ],
    }

    text_document = TextDocumentWithLabeledSpansAndBinaryRelations(
        id=document.pmid, text=document.sentence, metadata=metadata
    )

    gene = LabeledSpan(
        start=document.gene[0].start,
        end=document.gene[0].end,
        label="GENE",
    )
    text_document.labeled_spans.append(gene)

    cancer = LabeledSpan(
        start=document.cancer[0].start,
        end=document.cancer[0].end,
        label="CANCER",
    )
    text_document.labeled_spans.append(cancer)

    label = get_relation_label(
        cge=document.cge, ccs=document.ccs, ige=document.ige, pt=document.pt
    )

    if label is not None:
        relation = BinaryRelation(
            head=gene,
            tail=cancer,
            label=label,
        )
        text_document.binary_relations.append(relation)

    return text_document


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

    DOCUMENT_CONVERTERS = {
        TextDocumentWithLabeledSpansAndBinaryRelations: convert_to_text_document_with_labeled_spans_and_binary_relations
    }

    def _generate_document(self, example, **kwargs):
        return example_to_document(example)

    def _generate_example(self, document: ComagcDocument, **kwargs) -> Dict[str, Any]:
        return document_to_example(document)


def get_relation_label(cge: str, ccs: str, pt: str, ige: str) -> Optional[str]:
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
            "CCS": "cancerTOcancer",
            "IGE": "up-regulated",
            "PT": "observation",
            "Gene class": "biomarker",
        },
        {
            "CGE": "increased",
            "CCS": "cancerTOcancer",
            "IGE": "down-regulated",
            "PT": "observation",
            "Gene class": "biomarker",
        },
    ]

    for rule in rules:
        if (
            (rule["CGE"] == "*" or cge == rule["CGE"])
            and (rule["CCS"] == "*" or ccs == rule["CCS"])
            and (rule["IGE"] == "*" or ige == rule["IGE"])
            and (rule["PT"] == "*" or pt == rule["PT"])
        ):
            return rule["Gene class"]

    # Commented out to avoid spamming the logs
    # logger.warning("No rule matched. cge: " + cge + " - ccs: " + ccs + " - ige: " + ige + " - pt: " + pt)
    # NOTE: In case no inference rule is applicable, no relation is returned and
    #       eventually no relation is added to the document.
    return None
