from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import datasets
from pie_modules.document.processing import token_based_document_to_text_based
from pytorch_ie.annotations import BinaryRelation, LabeledSpan
from pytorch_ie.core import Annotation, AnnotationList, annotation_field
from pytorch_ie.documents import (
    TextDocumentWithLabeledSpansAndBinaryRelations,
    TokenBasedDocument,
)

from pie_datasets import GeneratorBasedBuilder


@dataclass(eq=True, frozen=True)
class TokenRelation(Annotation):
    head_idx: int
    tail_idx: int
    label: str
    score: float = field(default=1.0, compare=False)


@dataclass(eq=True, frozen=True)
class TokenAttribute(Annotation):
    idx: int
    label: str


@dataclass
class TacredDocument(TokenBasedDocument):
    stanford_ner: AnnotationList[TokenAttribute] = annotation_field(target="tokens")
    stanford_pos: AnnotationList[TokenAttribute] = annotation_field(target="tokens")
    entities: AnnotationList[LabeledSpan] = annotation_field(target="tokens")
    relations: AnnotationList[BinaryRelation] = annotation_field(target="entities")
    dependency_relations: AnnotationList[TokenRelation] = annotation_field(target="tokens")


@dataclass
class SimpleTacredDocument(TokenBasedDocument):
    labeled_spans: AnnotationList[LabeledSpan] = annotation_field(target="tokens")
    binary_relations: AnnotationList[BinaryRelation] = annotation_field(target="labeled_spans")


def example_to_document(
    example: Dict[str, Any],
    relation_labels: datasets.ClassLabel,
    ner_labels: datasets.ClassLabel,
) -> TacredDocument:
    document = TacredDocument(
        tokens=tuple(example["token"]), id=example["id"], metadata=dict(doc_id=example["docid"])
    )

    for idx, (ner, pos) in enumerate(zip(example["stanford_ner"], example["stanford_pos"])):
        document.stanford_ner.append(TokenAttribute(idx=idx, label=ner))
        document.stanford_pos.append(TokenAttribute(idx=idx, label=pos))

    for tail_idx, (deprel_label, head_idx) in enumerate(
        zip(example["stanford_deprel"], example["stanford_head"])
    ):
        if head_idx >= 0:
            document.dependency_relations.append(
                TokenRelation(
                    head_idx=head_idx,
                    tail_idx=tail_idx,
                    label=deprel_label,
                )
            )

    head = LabeledSpan(
        start=example["subj_start"],
        end=example["subj_end"],
        label=ner_labels.int2str(example["subj_type"]),
    )
    tail = LabeledSpan(
        start=example["obj_start"],
        end=example["obj_end"],
        label=ner_labels.int2str(example["obj_type"]),
    )
    document.entities.append(head)
    document.entities.append(tail)

    relation_str = relation_labels.int2str(example["relation"])
    relation = BinaryRelation(head=head, tail=tail, label=relation_str)
    document.relations.append(relation)

    return document


def _entity_to_dict(
    entity: LabeledSpan, key_prefix: str = "", labels: Optional[datasets.ClassLabel] = None
) -> Dict[str, Any]:
    return {
        f"{key_prefix}start": entity.start,
        f"{key_prefix}end": entity.end,
        f"{key_prefix}type": labels.str2int(entity.label) if labels is not None else entity.label,
    }


def document_to_example(
    document: TacredDocument,
    ner_labels: Optional[datasets.ClassLabel] = None,
    relation_labels: Optional[datasets.ClassLabel] = None,
) -> Dict[str, Any]:
    token = list(document.tokens)
    stanford_ner_dict = {ner.idx: ner.label for ner in document.stanford_ner}
    stanford_pos_dict = {pos.idx: pos.label for pos in document.stanford_pos}
    stanford_ner = [stanford_ner_dict[idx] for idx in range(len(token))]
    stanford_pos = [stanford_pos_dict[idx] for idx in range(len(token))]

    stanford_deprel = ["ROOT"] * len(document.tokens)
    stanford_head = [-1] * len(document.tokens)
    for dep_rel in document.dependency_relations:
        stanford_deprel[dep_rel.tail_idx] = dep_rel.label
        stanford_head[dep_rel.tail_idx] = dep_rel.head_idx

    rel = document.relations[0]
    obj: LabeledSpan = rel.tail
    subj: LabeledSpan = rel.head
    return {
        "id": document.id,
        "docid": document.metadata["doc_id"],
        "relation": rel.label if relation_labels is None else relation_labels.str2int(rel.label),
        "token": token,
        "stanford_ner": stanford_ner,
        "stanford_pos": stanford_pos,
        "stanford_deprel": stanford_deprel,
        "stanford_head": stanford_head,
        **_entity_to_dict(obj, key_prefix="obj_", labels=ner_labels),
        **_entity_to_dict(subj, key_prefix="subj_", labels=ner_labels),
    }


def convert_to_text_document_with_labeled_spans_and_binary_relations(
    document: TacredDocument,
) -> TextDocumentWithLabeledSpansAndBinaryRelations:
    doc_simplified = document.as_type(
        SimpleTacredDocument,
        field_mapping={"entities": "labeled_spans", "relations": "binary_relations"},
    )
    result = token_based_document_to_text_based(
        doc_simplified,
        result_document_type=TextDocumentWithLabeledSpansAndBinaryRelations,
        join_tokens_with=" ",
    )
    return result


class TacredConfig(datasets.BuilderConfig):
    """BuilderConfig for Tacred."""

    def __init__(self, **kwargs):
        """BuilderConfig for Tacred.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super().__init__(**kwargs)


class Tacred(GeneratorBasedBuilder):
    DOCUMENT_TYPE = TacredDocument

    DOCUMENT_CONVERTERS = {
        TextDocumentWithLabeledSpansAndBinaryRelations: convert_to_text_document_with_labeled_spans_and_binary_relations,
    }

    BASE_DATASET_PATH = "DFKI-SLT/tacred"
    BASE_DATASET_REVISION = "c801dc186b40a532c5820b4662570390da90431b"

    BUILDER_CONFIGS = [
        TacredConfig(
            name="original", version=datasets.Version("1.1.0"), description="The original TACRED."
        ),
        TacredConfig(
            name="revisited",
            version=datasets.Version("1.1.0"),
            description="The revised TACRED (corrected labels in dev and test split).",
        ),
        TacredConfig(
            name="re-tacred",
            version=datasets.Version("1.1.0"),
            description="Relabeled TACRED (corrected labels for all splits and pruned)",
        ),
    ]

    def _generate_document_kwargs(self, dataset):
        return {
            "ner_labels": dataset.features["subj_type"],
            "relation_labels": dataset.features["relation"],
        }

    def _generate_document(self, example, **kwargs):
        return example_to_document(example, **kwargs)
