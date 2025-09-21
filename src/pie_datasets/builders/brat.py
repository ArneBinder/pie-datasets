import dataclasses
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import datasets
from pie_core import Annotation, AnnotationLayer, Document, annotation_field

from pie_datasets import GeneratorBasedBuilder

logger = logging.getLogger(__name__)


def _post_init_single_label(self):
    if not isinstance(self.label, str):
        raise ValueError("label must be a single string.")

    if not isinstance(self.score, float):
        raise ValueError("score must be a single float.")


@dataclasses.dataclass(eq=True, frozen=True)
class BratRelation(Annotation):
    head: Annotation
    tail: Annotation
    label: str
    score: float = dataclasses.field(default=1.0, compare=False)

    def __post_init__(self) -> None:
        _post_init_single_label(self)

    def resolve(self) -> Any:
        return self.label, (self.head.resolve(), self.tail.resolve())


@dataclasses.dataclass(eq=True, frozen=True)
class BratSpan(Annotation):
    start: int
    end: int
    label: str
    score: float = dataclasses.field(default=1.0, compare=False)

    def __post_init__(self) -> None:
        _post_init_single_label(self)

    def __str__(self) -> str:
        if not self.is_attached:
            return super().__str__()
        return str(self.target[self.start : self.end])

    def resolve(self) -> Any:
        if self.is_attached:
            return self.label, self.target[self.start : self.end]
        else:
            raise ValueError(f"{self} is not attached to a target.")


@dataclasses.dataclass(eq=True, frozen=True)
class BratMultiSpan(Annotation):
    slices: Tuple[Tuple[int, int], ...]
    label: str
    score: float = dataclasses.field(default=1.0, compare=False)

    def __post_init__(self) -> None:
        if isinstance(self.slices, list):
            object.__setattr__(self, "slices", tuple(tuple(s) for s in self.slices))
        _post_init_single_label(self)

    def __str__(self) -> str:
        if not self.is_attached:
            return super().__str__()
        return str(tuple(self.target[start:end] for start, end in self.slices))

    def resolve(self) -> Any:
        if self.is_attached:
            return self.label, tuple(self.target[start:end] for start, end in self.slices)
        else:
            raise ValueError(f"{self} is not attached to a target.")


@dataclasses.dataclass(eq=True, frozen=True)
class BratAttribute(Annotation):
    annotation: Annotation
    label: str
    value: Optional[str] = None
    score: Optional[float] = dataclasses.field(default=None, compare=False)

    def resolve(self) -> Any:
        value = self.value if self.value is not None else True
        return value, self.label, self.annotation.resolve()


@dataclasses.dataclass(eq=True, frozen=True)
class BratNote(Annotation):
    annotation: Annotation
    label: str
    value: str

    def resolve(self) -> Any:
        return self.value, self.label, self.annotation.resolve()


@dataclasses.dataclass
class BaseBratDocument(Document):
    text: str
    id: Optional[str] = None
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class BratDocument(BaseBratDocument):
    spans: AnnotationLayer[BratMultiSpan] = annotation_field(target="text")
    relations: AnnotationLayer[BratRelation] = annotation_field(target="spans")
    attributes: AnnotationLayer[BratAttribute] = annotation_field(targets=["spans", "relations"])
    notes: AnnotationLayer[BratNote] = annotation_field(
        targets=["spans", "relations", "attributes"]
    )


@dataclasses.dataclass
class BratDocumentWithMergedSpans(BaseBratDocument):
    spans: AnnotationLayer[BratSpan] = annotation_field(target="text")
    relations: AnnotationLayer[BratRelation] = annotation_field(target="spans")
    attributes: AnnotationLayer[BratAttribute] = annotation_field(targets=["spans", "relations"])
    notes: AnnotationLayer[BratNote] = annotation_field(
        targets=["spans", "relations", "attributes"]
    )


def dl2ld(dict_of_lists: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    return [dict(zip(dict_of_lists, t)) for t in zip(*dict_of_lists.values())]


def ld2dl(
    list_fo_dicts: List[Dict[str, Any]], keys: Optional[List[str]] = None
) -> Dict[str, List[Any]]:
    keys = keys or list(list_fo_dicts[0])
    return {k: [dic[k] for dic in list_fo_dicts] for k in keys}


def example_to_document(
    example: Dict[str, Any], merge_fragmented_spans: bool = False
) -> Union[BratDocument, BratDocumentWithMergedSpans]:
    if merge_fragmented_spans:
        doc = BratDocumentWithMergedSpans(text=example["context"], id=example["file_name"])
    else:
        doc = BratDocument(text=example["context"], id=example["file_name"])

    id2annotation: Dict[str, Annotation] = dict()

    # create span annotations
    spans: Dict[str, BratSpan] = dict()
    doc.metadata["span_ids"] = []
    doc.metadata["span_locations"] = []
    doc.metadata["span_texts"] = []
    for span_dict in dl2ld(example["spans"]):
        starts: List[int] = span_dict["locations"]["start"]
        ends: List[int] = span_dict["locations"]["end"]
        slices = tuple(zip(starts, ends))
        # sanity check
        span_text_parts = [doc.text[start:end] for start, end in slices]
        joined_span_texts_stripped = " ".join(span_text_parts).strip()
        span_text_stripped = span_dict["text"].strip()
        if joined_span_texts_stripped != span_text_stripped:
            logger.warning(
                f"joined span parts do not match stripped span text field content. "
                f'joined_span_texts_stripped: "{joined_span_texts_stripped}" != stripped "text": "{span_text_stripped}"'
            )
        if merge_fragmented_spans:
            if len(starts) > 1:
                # check if the text in between the fragments holds only space
                merged_content_texts = [
                    doc.text[start:end] for start, end in zip(ends[:-1], starts[1:])
                ]
                merged_content_texts_not_empty = [
                    text.strip() for text in merged_content_texts if text.strip() != ""
                ]
                if len(merged_content_texts_not_empty) > 0:
                    logger.warning(
                        f"document '{doc.id}' contains a non-contiguous span with text content in between "
                        f"(will be merged into a single span): "
                        f"newly covered text parts: {merged_content_texts_not_empty}, "
                        f"merged span text: '{doc.text[starts[0]:ends[-1]]}', "
                        f"annotation: {span_dict}"
                    )
            # just take everything
            start = min(starts)
            end = max(ends)
            span = BratSpan(start=start, end=end, label=span_dict["type"])
        else:
            span = BratMultiSpan(slices=slices, label=span_dict["type"])
        spans[span_dict["id"]] = span

        # add span annotation to the document
        doc.spans.append(span)
        doc.metadata["span_ids"].append(span_dict["id"])
        doc.metadata["span_locations"].append(slices)
        doc.metadata["span_texts"].append(span_dict["text"])

        id2annotation[span_dict["id"]] = span

    # create relation annotations
    relations: Dict[str, BratRelation] = dict()
    doc.metadata["relation_ids"] = []
    for rel_dict in dl2ld(example["relations"]):
        arguments = dict(zip(rel_dict["arguments"]["type"], rel_dict["arguments"]["target"]))
        assert set(arguments) == {"Arg1", "Arg2"}
        head = spans[arguments["Arg1"]]
        tail = spans[arguments["Arg2"]]
        rel = BratRelation(head=head, tail=tail, label=rel_dict["type"])
        relations[rel_dict["id"]] = rel

        # add relation annotation to the document
        doc.relations.append(rel)
        doc.metadata["relation_ids"].append(rel_dict["id"])

        id2annotation[rel_dict["id"]] = rel

    # create equivalence relation annotations
    equivalence_relations = dl2ld(example["equivalence_relations"])
    if len(equivalence_relations) > 0:
        raise NotImplementedError("converting equivalence_relations is not yet implemented")

    # create event annotations
    events = dl2ld(example["events"])
    if len(events) > 0:
        raise NotImplementedError("converting events is not yet implemented")

    # create attribute annotations
    attributes: Dict[str, BratAttribute] = dict()
    doc.metadata["attribute_ids"] = []
    attribute_annotation2dict: Dict[BratAttribute, Dict[str, Any]] = defaultdict(dict)
    for attribute_dict in dl2ld(example["attributions"]):
        target_id = attribute_dict["target"]
        if target_id not in id2annotation:
            raise Exception(
                f"attribute target annotation {target_id} not found in any of the target layers (spans, relations)"
            )
        target_annotation = id2annotation[target_id]
        attribute = BratAttribute(
            annotation=target_annotation,
            label=attribute_dict["type"],
            value=attribute_dict["value"],
        )
        attributes[attribute_dict["id"]] = attribute

        # add attribute annotation to the document
        doc.attributes.append(attribute)
        doc.metadata["attribute_ids"].append(attribute_dict["id"])

        id2annotation[attribute_dict["id"]] = attribute

        # check for duplicates: this works only because the attribute was already added to the document
        # (annotations attached to different documents are never equal)
        if attribute in attribute_annotation2dict:
            prev_attribute = attribute_annotation2dict[attribute]
            logger.warning(
                f"document {doc.id}: annotation exists twice: {prev_attribute['id']} and {attribute_dict['id']} "
                f"are identical"
            )
        attribute_annotation2dict[attribute] = attribute_dict

    # create normalization annotations
    normalizations = dl2ld(example["normalizations"])
    if len(normalizations) > 0:
        raise NotImplementedError("converting normalizations is not yet implemented")

    # create note annotations
    notes: Dict[str, BratNote] = dict()
    doc.metadata["note_ids"] = []
    note_annotation2dict: Dict[Annotation, Dict[str, Any]] = dict()
    for note_dict in dl2ld(example["notes"]):
        target_id = note_dict["target"]
        if target_id not in id2annotation:
            raise Exception(
                f"note target annotation {target_id} not found in any of the target layers (spans, relations, "
                f"attributes)"
            )
        annotation = id2annotation[target_id]
        note = BratNote(
            annotation=annotation,
            label=note_dict["type"],
            value=note_dict["note"],
        )
        notes[note_dict["id"]] = note

        # add note annotation to the document
        doc.notes.append(note)
        doc.metadata["note_ids"].append(note_dict["id"])

        id2annotation[note_dict["id"]] = note

        # check for duplicates: this works only because the note was already added to the document
        # (annotations attached to different documents are never equal)
        if note in note_annotation2dict:
            prev_note = note_annotation2dict[note]
            logger.warning(
                f"document {doc.id}: annotation exists twice: {prev_note['id']} and {note_dict['id']} "
                f"are identical"
            )
        note_annotation2dict[note] = note_dict

    return doc


def document_to_example(
    document: Union[BratDocument, BratDocumentWithMergedSpans]
) -> Dict[str, Any]:
    annotation2id: Dict[Annotation, str] = dict()
    example: Dict[str, Any] = {
        "context": document.text,
        "file_name": document.id,
    }
    span_dicts: Dict[Union[BratSpan, BratMultiSpan], Dict[str, Any]] = dict()
    assert len(document.metadata["span_locations"]) == len(document.spans)
    assert len(document.metadata["span_texts"]) == len(document.spans)
    assert len(document.metadata["span_ids"]) == len(document.spans)
    for i, span in enumerate(document.spans):
        span_id = document.metadata["span_ids"][i]
        annotation2id[span] = span_id
        locations = tuple((start, end) for start, end in document.metadata["span_locations"][i])
        if isinstance(span, BratSpan):
            assert locations[0][0] == span.start
            assert locations[-1][1] == span.end
        elif isinstance(span, BratMultiSpan):
            assert span.slices == locations
        else:
            raise TypeError(f"span has unknown type [{type(span)}]: {span}")

        starts, ends = zip(*locations)
        span_dict = {
            "id": document.metadata["span_ids"][i],
            "locations": {
                "start": list(starts),
                "end": list(ends),
            },
            "text": document.metadata["span_texts"][i],
            "type": span.label,
        }
        if span in span_dicts:
            prev_ann_dict = span_dicts[span]
            ann_dict = span_dict
            logger.warning(
                f"document {document.id}: annotation exists twice: {prev_ann_dict['id']} and {ann_dict['id']} "
                f"are identical"
            )
        span_dicts[span] = span_dict
    example["spans"] = ld2dl(list(span_dicts.values()), keys=["id", "type", "locations", "text"])

    relation_dicts: Dict[BratRelation, Dict[str, Any]] = dict()
    assert len(document.metadata["relation_ids"]) == len(document.relations)
    for i, rel in enumerate(document.relations):
        rel_id = document.metadata["relation_ids"][i]
        annotation2id[rel] = rel_id
        arg1_id = span_dicts[rel.head]["id"]
        arg2_id = span_dicts[rel.tail]["id"]
        relation_dict = {
            "id": rel_id,
            "type": rel.label,
            "arguments": {
                "type": ["Arg1", "Arg2"],
                "target": [arg1_id, arg2_id],
            },
        }
        if rel in relation_dicts:
            prev_ann_dict = relation_dicts[rel]
            ann_dict = relation_dict
            logger.warning(
                f"document {document.id}: annotation exists twice: {prev_ann_dict['id']} and {ann_dict['id']} "
                f"are identical"
            )
        relation_dicts[rel] = relation_dict

    example["relations"] = ld2dl(list(relation_dicts.values()), keys=["id", "type", "arguments"])

    example["equivalence_relations"] = ld2dl([], keys=["type", "targets"])
    example["events"] = ld2dl([], keys=["id", "type", "trigger", "arguments"])

    attribute_dicts: Dict[Annotation, Dict[str, Any]] = dict()
    for i, attribute_annotation in enumerate(document.attributes):
        attribute_id = document.metadata["attribute_ids"][i]
        annotation2id[attribute_annotation] = attribute_id
        target_id = annotation2id[attribute_annotation.annotation]
        attribute_dict = {
            "id": attribute_id,
            "type": attribute_annotation.label,
            "target": target_id,
            "value": attribute_annotation.value,
        }
        if attribute_annotation in attribute_dicts:
            prev_ann_dict = attribute_dicts[attribute_annotation]
            logger.warning(
                f"document {document.id}: annotation exists twice: {prev_ann_dict['id']} and {attribute_dict['id']} "
                f"are identical"
            )
        attribute_dicts[attribute_annotation] = attribute_dict
    example["attributions"] = ld2dl(
        list(attribute_dicts.values()), keys=["id", "type", "target", "value"]
    )

    example["normalizations"] = ld2dl(
        [], keys=["id", "type", "target", "resource_id", "entity_id"]
    )

    notes_dicts: Dict[Annotation, Dict[str, Any]] = dict()
    for i, note_annotation in enumerate(document.notes):
        note_id = document.metadata["note_ids"][i]
        annotation2id[note_annotation] = note_id
        target_id = annotation2id[note_annotation.annotation]
        note_dict = {
            "id": note_id,
            "type": note_annotation.label,
            "target": target_id,
            "note": note_annotation.value,
        }
        if note_annotation in notes_dicts:
            prev_ann_dict = notes_dicts[note_annotation]
            logger.warning(
                f"document {document.id}: annotation exists twice: {prev_ann_dict['id']} and {note_dict['id']} "
                f"are identical"
            )
        notes_dicts[note_annotation] = note_dict
    example["notes"] = ld2dl(list(notes_dicts.values()), keys=["id", "type", "target", "note"])

    return example


class BratConfig(datasets.BuilderConfig):
    """BuilderConfig for BratDatasetLoader."""

    def __init__(self, merge_fragmented_spans: bool = False, **kwargs):
        """BuilderConfig for DocRED.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super().__init__(**kwargs)
        self.merge_fragmented_spans = merge_fragmented_spans


class BratBuilder(GeneratorBasedBuilder):
    DOCUMENT_TYPES = {
        "default": BratDocument,
        "merge_fragmented_spans": BratDocumentWithMergedSpans,
    }

    DEFAULT_CONFIG_NAME = "default"
    BUILDER_CONFIGS = [
        BratConfig(name="default"),
        BratConfig(name="merge_fragmented_spans", merge_fragmented_spans=True),
    ]

    BASE_DATASET_PATH = "DFKI-SLT/brat"
    BASE_DATASET_REVISION = (
        None  # it is highly recommended to set this to a commit hash in any derived builder
    )

    def _generate_document(self, example, **kwargs):
        return example_to_document(
            example, merge_fragmented_spans=self.config.merge_fragmented_spans
        )

    def _generate_example(self, document: Document, **kwargs) -> Dict[str, Any]:
        if not isinstance(document, (BratDocument, BratDocumentWithMergedSpans)):
            raise TypeError(f"document type {type(document)} is not supported")
        return document_to_example(document)
