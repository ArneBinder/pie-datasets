from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import datasets
import pytorch_ie.data.builder
from pytorch_ie import token_based_document_to_text_based
from pytorch_ie.annotations import BinaryRelation, LabeledSpan, _post_init_single_label
from pytorch_ie.core import Annotation, AnnotationList, Document, annotation_field
from pytorch_ie.documents import (
    TextDocumentWithLabeledSpansAndBinaryRelations,
    TokenBasedDocument,
)


@dataclass(eq=True, frozen=True)
class TokenRelation(Annotation):
    head_idx: int
    tail_idx: int
    label: str
    score: float = 1.0

    def __post_init__(self) -> None:
        _post_init_single_label(self)


@dataclass(eq=True, frozen=True)
class TokenAttribute(Annotation):
    idx: int
    label: str


@dataclass
class TacredDocument(Document):
    tokens: Tuple[str, ...]
    id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
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
    relation_int2str: Callable[[int], str],
    ner_int2str: Callable[[int], str],
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
        label=ner_int2str(example["subj_type"]),
    )
    tail = LabeledSpan(
        start=example["obj_start"],
        end=example["obj_end"],
        label=ner_int2str(example["obj_type"]),
    )
    document.entities.append(head)
    document.entities.append(tail)

    relation_str = relation_int2str(example["relation"])
    relation = BinaryRelation(head=head, tail=tail, label=relation_str)
    document.relations.append(relation)

    return document


def _entity_to_dict(
    entity: LabeledSpan, key_prefix: str = "", label_mapping: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    return {
        f"{key_prefix}start": entity.start,
        f"{key_prefix}end": entity.end,
        f"{key_prefix}type": label_mapping[entity.label]
        if label_mapping is not None
        else entity.label,
    }


def document_to_example(
    document: TacredDocument,
    ner_names: Optional[List[str]] = None,
    relation_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    ner2idx = {name: idx for idx, name in enumerate(ner_names)} if ner_names is not None else None
    rel2idx = (
        {name: idx for idx, name in enumerate(relation_names)}
        if relation_names is not None
        else None
    )

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
        "relation": rel.label if rel2idx is None else rel2idx[rel.label],
        "token": token,
        "stanford_ner": stanford_ner,
        "stanford_pos": stanford_pos,
        "stanford_deprel": stanford_deprel,
        "stanford_head": stanford_head,
        **_entity_to_dict(obj, key_prefix="obj_", label_mapping=ner2idx),
        **_entity_to_dict(subj, key_prefix="subj_", label_mapping=ner2idx),
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


class Tacred(pytorch_ie.data.builder.GeneratorBasedBuilder):
    DOCUMENT_TYPE = TacredDocument

    DOCUMENT_CONVERTERS = {
        TextDocumentWithLabeledSpansAndBinaryRelations: convert_to_text_document_with_labeled_spans_and_binary_relations,
    }

    BASE_DATASET_PATH = "DFKI-SLT/tacred"

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
            "ner_int2str": dataset.features["subj_type"].int2str,
            "relation_int2str": dataset.features["relation"].int2str,
        }

    def _generate_document(self, example, **kwargs):
        return example_to_document(example, **kwargs)
