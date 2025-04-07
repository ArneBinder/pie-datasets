import dataclasses
import logging
from typing import Any, Dict

import datasets
from pie_core import AnnotationLayer, annotation_field
from pie_modules.annotations import BinaryRelation, LabeledSpan
from pie_modules.document.processing import token_based_document_to_text_based
from pie_modules.utils.sequence_tagging import (
    tag_sequence_to_token_spans,
    token_spans_to_tag_sequence,
)
from pie_modules.documents import (
    TextDocumentWithLabeledSpansAndBinaryRelations,
    TokenBasedDocument,
)

from pie_datasets import GeneratorBasedBuilder

log = logging.getLogger(__name__)


@dataclasses.dataclass
class SciDTBArgminDocument(TokenBasedDocument):
    units: AnnotationLayer[LabeledSpan] = annotation_field(target="tokens")
    relations: AnnotationLayer[BinaryRelation] = annotation_field(target="units")


@dataclasses.dataclass
class SimplifiedSciDTBArgminDocument(TokenBasedDocument):
    labeled_spans: AnnotationLayer[LabeledSpan] = annotation_field(target="tokens")
    binary_relations: AnnotationLayer[BinaryRelation] = annotation_field(target="labeled_spans")


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
        tag_sequence_to_token_spans(tag_sequence), key=lambda label_and_span: label_and_span[1][0]
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

    tag_sequence = token_spans_to_tag_sequence(
        labeled_spans=[
            (label, (unit.start, unit.end)) for label, unit in zip(labels, document.units)
        ],
        base_sequence_length=len(document.tokens),
        coding_scheme="IOB2",
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
