from dataclasses import dataclass

from pytorch_ie import AnnotationLayer, annotation_field
from pytorch_ie.annotations import BinaryRelation, LabeledSpan
from pytorch_ie.documents import TextBasedDocument


@dataclass
class ComagcDocument(TextBasedDocument):
    entities: AnnotationLayer[LabeledSpan] = annotation_field(target="text")
    relations: AnnotationLayer[BinaryRelation] = annotation_field(target="entities")


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
    )
    # entity name is the text of the entity (between the start and end positions)
    head = LabeledSpan(
        start=example["gene"]["pos"][0],
        end=example["gene"]["pos"][1] + 1,
        label="GENE",
    )
    tail = LabeledSpan(
        start=example["cancer"]["pos"][0],
        end=example["cancer"]["pos"][1] + 1,
        label="CANCER",
    )
    doc.entities.extend([head, tail])

    relation = BinaryRelation(head=head, tail=tail, label=get_relation(example))
    doc.relations.append(relation)
    return doc


def get_relation(example):
    """Simple rule-based function to determine the relation between the gene and the cancer."""
    # https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-14-323/tables/3

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

    return "unidentified"
