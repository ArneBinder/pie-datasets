import dataclasses
import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import datasets
from pytorch_ie.annotations import LabeledSpan, Span
from pytorch_ie.core import Annotation, AnnotationList, Document, annotation_field
from pytorch_ie.documents import TextDocumentWithLabeledSpans

from pie_datasets import ArrowBasedBuilder

logger = logging.getLogger(__name__)

VARIANT_ORIGINAL = "original"
VARIANT_REVISITED = "revisited"


@dataclasses.dataclass(eq=True, frozen=True)
class AnnotationCollection(Annotation):
    members: Tuple[Annotation, ...]
    idx: int  # dummy to enforce keeping duplicated entries


@dataclasses.dataclass(eq=True, frozen=True)
class BinaryRelationWithEvidence(Annotation):
    head: Annotation
    tail: Annotation
    label: str
    evidence: Tuple[Annotation, ...]
    score: float = 1.0


@dataclasses.dataclass
class DocredDocument(Document):
    title: str
    tokens: List[str] = dataclasses.field(default_factory=list)
    sentences: AnnotationList[Span] = annotation_field(target="tokens")
    entity_mentions: AnnotationList[LabeledSpan] = annotation_field(target="tokens")
    entities: AnnotationList[AnnotationCollection] = annotation_field(target="entity_mentions")
    relations: AnnotationList[BinaryRelationWithEvidence] = annotation_field(
        targets=["entities", "sentences"]
    )
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)


def dict_list2list_dict(dict_of_lists: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    return [dict(zip(dict_of_lists, t)) for t in zip(*dict_of_lists.values())]


def list_dict2dict_list(
    list_of_dicts: List[Dict[str, Any]], keys: Optional[List[str]] = None
) -> Dict[str, List[Any]]:
    if keys is None:
        if len(list_of_dicts) == 0:
            raise Exception("keys are required if list_of_dicts is an empty list")
        keys = list(list_of_dicts[0].keys())
    return {k: [dic[k] for dic in list_of_dicts] for k in keys}


def example2document(
    example: Dict[str, Any], variant: str, verbose: bool = False
) -> DocredDocument:
    document = DocredDocument(title=example["title"])
    start = 0
    for idx, sent in enumerate(example["sents"]):
        document.tokens.extend(sent)
        end = len(document.tokens)
        document.sentences.append(Span(start=start, end=end))
        start = end

    for idx, vertex in enumerate(example["vertexSet"]):
        current_entity_mentions: List[LabeledSpan] = []
        for mention_dict in vertex:
            target_sentence = document.sentences[mention_dict["sent_id"]]
            start_offset, end_offset = mention_dict["pos"]
            current_entity_mentions.append(
                LabeledSpan(
                    start=target_sentence.start + start_offset,
                    end=target_sentence.start + end_offset,
                    label=mention_dict["type"],
                )
            )
        document.entity_mentions.extend(current_entity_mentions)
        entity = AnnotationCollection(members=tuple(current_entity_mentions), idx=idx)
        document.entities.append(entity)

    if variant == VARIANT_ORIGINAL:
        label_dicts = dict_list2list_dict(example["labels"])
        key_head = "head"
        key_tail = "tail"
        key_relation = "relation_id"
    elif variant == VARIANT_REVISITED:
        label_dicts = example["labels"]
        key_head = "h"
        key_tail = "t"
        key_relation = "r"
    else:
        raise ValueError(
            f"variant has to be one of: {VARIANT_ORIGINAL} or {VARIANT_REVISITED}, but it is: {variant}"
        )
    for label_dict in label_dicts:
        head = document.entities[label_dict[key_head]]
        tail = document.entities[label_dict[key_tail]]
        label = label_dict[key_relation]
        if variant == VARIANT_ORIGINAL:
            label += f":{label_dict['relation_text']}"
        evidence = [document.sentences[sent_idx] for sent_idx in label_dict["evidence"]]
        relation = BinaryRelationWithEvidence(
            head=head, tail=tail, label=label, evidence=tuple(evidence)
        )
        document.relations.append(relation)

    return document


def _span_entries(span: Span) -> List[List[Any]]:
    return span.target[span.start : span.end]


def document2example(doc: DocredDocument, variant: str) -> Dict[str, Any]:
    sents = [_span_entries(sentence) for sentence in doc.sentences]

    entity2idx = {entity: idx for idx, entity in enumerate(doc.entities)}
    sentence2idx = {}
    token_idx2sent_idx_and_offset = {}
    token_idx = 0
    for sentence_idx, sentence in enumerate(doc.sentences):
        sentence2idx[sentence] = sentence_idx
        for current_token_idx, token in enumerate(_span_entries(sentence)):
            token_idx2sent_idx_and_offset[token_idx] = (sentence_idx, current_token_idx)
            token_idx += 1

    vertexSet = []
    for idx, entity in enumerate(doc.entities):
        entity_mention_dicts = []
        entity_mention: LabeledSpan
        for entity_mention in entity.members:
            start_sent_id, start_pos = token_idx2sent_idx_and_offset[entity_mention.start]
            end_sent_id_inclusive, end_pos_inclusive = token_idx2sent_idx_and_offset[
                entity_mention.end - 1
            ]
            assert start_sent_id == end_sent_id_inclusive
            entity_mention_dict = {
                # Do not set the entity string. It can not be correctly reconstructed from the tokens because we
                # do not know when to add whitespace.
                # "name": " ".join(_span_entries(entity_mention)),
                "name": None,
                "type": entity_mention.label,
                "sent_id": start_sent_id,
                "pos": [start_pos, end_pos_inclusive + 1],
            }
            entity_mention_dicts.append(entity_mention_dict)
        vertexSet.append(entity_mention_dicts)

    if variant == VARIANT_ORIGINAL:
        key_head = "head"
        key_tail = "tail"
        key_relation = "relation_id"
    elif variant == VARIANT_REVISITED:
        key_head = "h"
        key_tail = "t"
        key_relation = "r"
    else:
        raise ValueError(
            f"variant has to be one of: {VARIANT_ORIGINAL} or {VARIANT_REVISITED}, but it is: {variant}"
        )

    label_dicts = []
    for rel_idx, relation in enumerate(doc.relations):
        head_idx = entity2idx[relation.head]
        tail_idx = entity2idx[relation.tail]
        evidence_indices = [sentence2idx[sentence] for sentence in relation.evidence]
        label_dict = {
            key_head: head_idx,
            key_tail: tail_idx,
            "evidence": evidence_indices,
        }
        if variant == VARIANT_ORIGINAL:
            relation_id, relation_text = relation.label.split(":")
            label_dict["relation_text"] = relation_text
        else:
            relation_id = relation.label
        label_dict[key_relation] = relation_id

        label_dicts.append(label_dict)

    if variant == VARIANT_ORIGINAL:
        labels = list_dict2dict_list(
            list_of_dicts=label_dicts,
            keys=["head", "tail", "relation_id", "relation_text", "evidence"],
        )
    else:
        labels = label_dicts  # type: ignore
    return {
        "title": doc.title,
        "sents": sents,
        "vertexSet": vertexSet,
        "labels": labels,
    }


def convert_to_text_document_with_labeled_spans(
    doc: DocredDocument, token_separator: str = " ", skip_overlapping_entities: bool = True
) -> TextDocumentWithLabeledSpans:
    text = ""
    token_offsets = []
    for idx, token in enumerate(doc.tokens):
        start = len(text)
        text += token
        end = len(text)
        token_offsets.append((start, end))
        if idx < len(doc.tokens) - 1:
            text += token_separator

    new_doc = TextDocumentWithLabeledSpans(
        text=text, id=doc.title, metadata=deepcopy(doc.metadata)
    )

    dummy_tags = [None] * len(text)
    for idx, token_entity_mention in enumerate(doc.entity_mentions):
        start = token_offsets[token_entity_mention.start][0]
        end = token_offsets[token_entity_mention.end - 1][1]
        if len(dummy_tags[start:end]) != (end - start):
            raise Exception(
                f"entity mention out of boundaries: start={start}, end={end}, text length={len(text)}"
            )
        if skip_overlapping_entities and dummy_tags[start:end] != [None] * (end - start):
            logger.warning("span overlap detected, skip the entity")
        else:
            char_entity_mention = LabeledSpan(
                start=start, end=end, label=token_entity_mention.label
            )
            new_doc.labeled_spans.append(char_entity_mention)
            dummy_tags[start:end] = [token_entity_mention.label] * (end - start)

    return new_doc


class DocredConfig(datasets.BuilderConfig):
    """BuilderConfig for DocRED."""

    def __init__(self, **kwargs):
        """BuilderConfig for DocRED.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super().__init__(**kwargs)


class Docred(ArrowBasedBuilder):
    DOCUMENT_TYPE = DocredDocument

    DOCUMENT_CONVERTERS = {
        TextDocumentWithLabeledSpans: convert_to_text_document_with_labeled_spans
    }

    BASE_DATASET_PATH = "tonytan48/Re-DocRED"

    BUILDER_CONFIGS = [
        DocredConfig(
            name=VARIANT_REVISITED,
            version=datasets.Version("1.0.0"),
            description="The revisited DocRED dataset (Re-DocRED).",
        ),
    ]

    def _generate_document_kwargs(self, dataset):
        return {"variant": self.config.name}

    def _generate_document(self, example, variant, **kwargs):
        return example2document(example, variant)
