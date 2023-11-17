import dataclasses
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import datasets
from pytorch_ie.annotations import BinaryRelation, LabeledSpan
from pytorch_ie.core import AnnotationList, Document, annotation_field
from pytorch_ie.documents import (
    TextDocumentWithLabeledSpansAndBinaryRelations,
    TokenBasedDocument,
)
from pytorch_ie.utils.span import bio_tags_to_spans

from pie_datasets import GeneratorBasedBuilder
from pie_datasets.document.conversion import token_based_document_to_text_based

log = logging.getLogger(__name__)


def labels_and_spans_to_bio_tags(
    labels: List[str], spans: List[Tuple[int, int]], sequence_length: int
) -> List[str]:
    bio_tags = ["O"] * sequence_length
    for label, (start, end) in zip(labels, spans):
        bio_tags[start] = f"B-{label}"
        for i in range(start + 1, end):
            bio_tags[i] = f"I-{label}"
    return bio_tags


@dataclasses.dataclass
class SciDTBArgminDocument(TokenBasedDocument):
    units: AnnotationList[LabeledSpan] = annotation_field(target="tokens")
    relations: AnnotationList[BinaryRelation] = annotation_field(target="units")


@dataclasses.dataclass
class SimplifiedSciDTBArgminDocument(TokenBasedDocument):
    labeled_spans: AnnotationList[LabeledSpan] = annotation_field(target="tokens")
    binary_relations: AnnotationList[BinaryRelation] = annotation_field(target="labeled_spans")


def example_to_document(
    example: Dict[str, Any],
    unit_bio: datasets.ClassLabel,
    unit_label: datasets.ClassLabel,
    relation: datasets.ClassLabel,
):
    document = SciDTBArgminDocument(id=example["id"], tokens=tuple(example["data"]["token"]))
    bio_tags = unit_bio.int2str(example["data"]["unit-bio"])
    unit_labels = unit_label.int2str(example["data"]["unit-label"])
    roles = relation.int2str(example["data"]["role"])
    tag_sequence = [
        f"{bio}-{label}|{role}|{parent_offset}"
        for bio, label, role, parent_offset in zip(
            bio_tags, unit_labels, roles, example["data"]["parent-offset"]
        )
    ]
    spans_with_label = sorted(
        bio_tags_to_spans(tag_sequence), key=lambda label_and_span: label_and_span[1][0]
    )
    labels, spans = zip(*spans_with_label)
    span_unit_labels, span_roles, span_parent_offsets = zip(
        *[label.split("|") for label in labels]
    )

    units = [
        LabeledSpan(start=start, end=end + 1, label=label)
        for (start, end), label in zip(spans, span_unit_labels)
    ]
    document.units.extend(units)

    # The relation direction is as in "f{head} {relation_label} {tail}"
    relations = []
    for idx, parent_offset in enumerate(span_parent_offsets):
        if span_roles[idx] != "none":
            relations.append(
                BinaryRelation(
                    head=units[idx], tail=units[idx + int(parent_offset)], label=span_roles[idx]
                )
            )

    document.relations.extend(relations)

    return document


def document_to_example(
    document: SciDTBArgminDocument,
    unit_bio: datasets.ClassLabel,
    unit_label: datasets.ClassLabel,
    relation: datasets.ClassLabel,
) -> Dict[str, Any]:
    unit2idx = {unit: idx for idx, unit in enumerate(document.units)}
    unit2parent_relation = {relation.head: relation for relation in document.relations}

    unit_labels = [unit.label for unit in document.units]
    roles = [
        unit2parent_relation[unit].label if unit in unit2parent_relation else "none"
        for unit in document.units
    ]
    parent_offsets = [
        unit2idx[unit2parent_relation[unit].tail] - idx if unit in unit2parent_relation else 0
        for idx, unit in enumerate(document.units)
    ]
    labels = [
        f"{unit_label}-{role}-{parent_offset}"
        for unit_label, role, parent_offset in zip(unit_labels, roles, parent_offsets)
    ]

    tag_sequence = labels_and_spans_to_bio_tags(
        labels=labels,
        spans=[(unit.start, unit.end) for unit in document.units],
        sequence_length=len(document.tokens),
    )
    bio_tags, unit_labels, roles, parent_offsets = zip(
        *[tag.split("-", maxsplit=3) for tag in tag_sequence]
    )

    data = {
        "token": list(document.tokens),
        "unit-bio": unit_bio.str2int(bio_tags),
        "unit-label": unit_label.str2int(unit_labels),
        "role": relation.str2int(roles),
        "parent-offset": [int(idx_str) for idx_str in parent_offsets],
    }
    result = {"id": document.id, "data": data}
    return result


def convert_to_text_document_with_labeled_spans_and_binary_relations(
    document: SciDTBArgminDocument,
) -> TextDocumentWithLabeledSpansAndBinaryRelations:
    doc_simplified = document.as_type(
        SimplifiedSciDTBArgminDocument,
        field_mapping={"units": "labeled_spans", "relations": "binary_relations"},
    )
    result = token_based_document_to_text_based(
        doc_simplified,
        result_document_type=TextDocumentWithLabeledSpansAndBinaryRelations,
        join_tokens_with=" ",
    )
    return result


class SciDTBArgmin(GeneratorBasedBuilder):
    DOCUMENT_TYPE = SciDTBArgminDocument

    DOCUMENT_CONVERTERS = {
        TextDocumentWithLabeledSpansAndBinaryRelations: convert_to_text_document_with_labeled_spans_and_binary_relations
    }

    BASE_DATASET_PATH = "DFKI-SLT/scidtb_argmin"
    BASE_DATASET_REVISION = "8c02587edcb47ab5b102692bd10bfffd1844a09b"

    BUILDER_CONFIGS = [datasets.BuilderConfig(name="default")]

    DEFAULT_CONFIG_NAME = "default"

    def _generate_document_kwargs(self, dataset):
        return {
            "unit_bio": dataset.features["data"].feature["unit-bio"],
            "unit_label": dataset.features["data"].feature["unit-label"],
            "relation": dataset.features["data"].feature["role"],
        }

    def _generate_document(self, example, unit_bio, unit_label, relation):
        return example_to_document(
            example,
            unit_bio=unit_bio,
            unit_label=unit_label,
            relation=relation,
        )
