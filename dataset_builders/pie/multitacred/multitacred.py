import itertools
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import datasets
from pytorch_ie import token_based_document_to_text_based
from pytorch_ie.annotations import BinaryRelation, LabeledSpan
from pytorch_ie.core import AnnotationList, annotation_field
from pytorch_ie.documents import (
    TextDocumentWithLabeledSpansAndBinaryRelations,
    TokenBasedDocument,
)

from pie_datasets import GeneratorBasedBuilder

_BACKTRANSLATION_TEST_SPLIT = "backtranslated_test"

_RETACRED = "retacred"

_REVISITED = "revisited"

_ORIGINAL = "original"

_VERSION = datasets.Version("1.1.0")

_LANGS = [
    "ar",
    "de",
    "es",
    "fi",
    "fr",
    "hi",
    "hu",
    "ja",
    "pl",
    "ru",
    "tr",
    "zh",
]

_DESC_TEXTS = {
    _ORIGINAL: "The original TACRED.",
    _REVISITED: "TACRED Revisited (corrected labels for 5k most challenging examples in dev and test split).",
    _RETACRED: "Relabeled TACRED (corrected labels for all splits and pruned)",
}


@dataclass
class MultiTacredDocument(TokenBasedDocument):
    entities: AnnotationList[LabeledSpan] = annotation_field(target="tokens")
    relations: AnnotationList[BinaryRelation] = annotation_field(target="entities")


def example_to_document(
    example: Dict[str, Any],
    relation_int2str: Callable[[int], str],
    ner_int2str: Callable[[int], str],
) -> MultiTacredDocument:
    document = MultiTacredDocument(id=example["id"], tokens=tuple(example["token"]))

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
    document: MultiTacredDocument,
    ner_names: Optional[List[str]] = None,
    relation_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if len(document.relations) != 1:
        raise Exception(
            f"MultiTacredDocument should have exactly one relation annotation, but it has: "
            f"{MultiTacredDocument.relations}"
        )
    if len(document.entities) != 2:
        raise Exception(
            f"MultiTacredDocument should have exactly two entity annotation, but it has: {MultiTacredDocument.entities}"
        )

    ner2idx = {name: idx for idx, name in enumerate(ner_names)} if ner_names is not None else None
    rel2idx = (
        {name: idx for idx, name in enumerate(relation_names)}
        if relation_names is not None
        else None
    )

    rel = document.relations[0]
    obj: LabeledSpan = rel.tail
    subj: LabeledSpan = rel.head
    return {
        "id": document.id,
        "relation": rel.label if rel2idx is None else rel2idx[rel.label],
        "token": list(document.tokens),
        **_entity_to_dict(obj, key_prefix="obj_", label_mapping=ner2idx),
        **_entity_to_dict(subj, key_prefix="subj_", label_mapping=ner2idx),
    }


@dataclass
class SimpleMultiTacredDocument(TokenBasedDocument):
    labeled_spans: AnnotationList[LabeledSpan] = annotation_field(target="tokens")
    binary_relations: AnnotationList[BinaryRelation] = annotation_field(target="labeled_spans")


def convert_to_text_document_with_labeled_spans_and_binary_relations(
    document: MultiTacredDocument,
) -> TextDocumentWithLabeledSpansAndBinaryRelations:
    doc_simplified = document.as_type(
        SimpleMultiTacredDocument,
        field_mapping={"entities": "labeled_spans", "relations": "binary_relations"},
    )
    result = token_based_document_to_text_based(
        doc_simplified,
        result_document_type=TextDocumentWithLabeledSpansAndBinaryRelations,
        join_tokens_with=" ",
    )
    return result


class MultiTacredConfig(datasets.BuilderConfig):
    """BuilderConfig for MultiTacred."""

    def __init__(self, label_variant, language, **kwargs):
        """BuilderConfig for MultiTacred.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super().__init__(version=_VERSION, **kwargs)
        self.language = language
        self.label_variant = label_variant


class MultiTacred(GeneratorBasedBuilder):
    DOCUMENT_TYPE = MultiTacredDocument

    DOCUMENT_CONVERTERS = {
        TextDocumentWithLabeledSpansAndBinaryRelations: convert_to_text_document_with_labeled_spans_and_binary_relations
    }

    BASE_DATASET_PATH = "DFKI-SLT/multitacred"

    BUILDER_CONFIGS = [
        MultiTacredConfig(
            name=f"{label_variant}-{language}",
            language=language,
            label_variant=label_variant,
            description=f"{_DESC_TEXTS[label_variant]} examples in language '{language}'.",
        )
        for (language, label_variant) in itertools.product(
            _LANGS, [_ORIGINAL, _REVISITED, _RETACRED]
        )
    ]

    def _generate_document_kwargs(self, dataset):
        return {
            "ner_int2str": dataset.features["subj_type"].int2str,
            "relation_int2str": dataset.features["relation"].int2str,
        }

    def _generate_document(self, example, **kwargs):
        return example_to_document(example, **kwargs)
