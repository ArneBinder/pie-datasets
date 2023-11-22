import dataclasses
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import datasets
from pytorch_ie.annotations import BinaryRelation, LabeledMultiSpan, LabeledSpan
from pytorch_ie.core import Annotation, AnnotationList, annotation_field
from pytorch_ie.documents import TextBasedDocument

from pie_datasets import GeneratorBasedBuilder

logger = logging.getLogger(__name__)


@dataclasses.dataclass(eq=True, frozen=True)
class BratAttribute(Annotation):
    annotation: Annotation
    label: str
    value: Optional[str] = None
    score: Optional[float] = dataclasses.field(default=None, compare=False)


@dataclasses.dataclass
class BratDocument(TextBasedDocument):
    spans: AnnotationList[LabeledMultiSpan] = annotation_field(target="text")
    relations: AnnotationList[BinaryRelation] = annotation_field(target="spans")
    span_attributes: AnnotationList[BratAttribute] = annotation_field(target="spans")
    relation_attributes: AnnotationList[BratAttribute] = annotation_field(target="relations")


@dataclasses.dataclass
class BratDocumentWithMergedSpans(TextBasedDocument):
    spans: AnnotationList[LabeledSpan] = annotation_field(target="text")
    relations: AnnotationList[BinaryRelation] = annotation_field(target="spans")
    span_attributes: AnnotationList[BratAttribute] = annotation_field(target="spans")
    relation_attributes: AnnotationList[BratAttribute] = annotation_field(target="relations")


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

    spans: Dict[str, LabeledSpan] = dict()
    span_locations: List[Tuple[Tuple[int, int], ...]] = []
    span_texts: List[str] = []
    for span_dict in dl2ld(example["spans"]):
        starts: List[int] = span_dict["locations"]["start"]
        ends: List[int] = span_dict["locations"]["end"]
        slices = tuple(zip(starts, ends))
        span_locations.append(slices)
        span_texts.append(span_dict["text"])
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
            span = LabeledSpan(start=start, end=end, label=span_dict["type"])
        else:
            span = LabeledMultiSpan(slices=slices, label=span_dict["type"])
        spans[span_dict["id"]] = span

    doc.spans.extend(spans.values())
    doc.metadata["span_ids"] = list(spans.keys())
    doc.metadata["span_locations"] = span_locations
    doc.metadata["span_texts"] = span_texts

    relations: Dict[str, BinaryRelation] = dict()
    for rel_dict in dl2ld(example["relations"]):
        arguments = dict(zip(rel_dict["arguments"]["type"], rel_dict["arguments"]["target"]))
        assert set(arguments) == {"Arg1", "Arg2"}
        head = spans[arguments["Arg1"]]
        tail = spans[arguments["Arg2"]]
        rel = BinaryRelation(head=head, tail=tail, label=rel_dict["type"])
        relations[rel_dict["id"]] = rel

    doc.relations.extend(relations.values())
    doc.metadata["relation_ids"] = list(relations.keys())

    equivalence_relations = dl2ld(example["equivalence_relations"])
    if len(equivalence_relations) > 0:
        raise NotImplementedError("converting equivalence_relations is not yet implemented")

    events = dl2ld(example["events"])
    if len(events) > 0:
        raise NotImplementedError("converting events is not yet implemented")

    attribute_annotations: Dict[str, Dict[str, BratAttribute]] = defaultdict(dict)
    attribute_ids = []
    for attribute_dict in dl2ld(example["attributions"]):
        target_id = attribute_dict["target"]
        if target_id in spans:
            target_layer_name = "spans"
            annotation = spans[target_id]
        elif target_id in relations:
            target_layer_name = "relations"
            annotation = relations[target_id]
        else:
            raise Exception("only span and relation attributes are supported yet")
        attribute = BratAttribute(
            annotation=annotation,
            label=attribute_dict["type"],
            value=attribute_dict["value"],
        )
        attribute_annotations[target_layer_name][attribute_dict["id"]] = attribute
        attribute_ids.append((target_layer_name, attribute_dict["id"]))

    doc.span_attributes.extend(attribute_annotations["spans"].values())
    doc.relation_attributes.extend(attribute_annotations["relations"].values())
    doc.metadata["attribute_ids"] = attribute_ids

    normalizations = dl2ld(example["normalizations"])
    if len(normalizations) > 0:
        raise NotImplementedError("converting normalizations is not yet implemented")

    notes = dl2ld(example["notes"])
    if len(notes) > 0:
        raise NotImplementedError("converting notes is not yet implemented")

    return doc


def document_to_example(
    document: Union[BratDocument, BratDocumentWithMergedSpans]
) -> Dict[str, Any]:
    example = {
        "context": document.text,
        "file_name": document.id,
    }
    span_dicts: Dict[Union[LabeledSpan, LabeledMultiSpan], Dict[str, Any]] = dict()
    assert len(document.metadata["span_locations"]) == len(document.spans)
    assert len(document.metadata["span_texts"]) == len(document.spans)
    assert len(document.metadata["span_ids"]) == len(document.spans)
    for i, span in enumerate(document.spans):
        locations = tuple((start, end) for start, end in document.metadata["span_locations"][i])
        if isinstance(span, LabeledSpan):
            assert locations[0][0] == span.start
            assert locations[-1][1] == span.end
        elif isinstance(span, LabeledMultiSpan):
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

    relation_dicts: Dict[BinaryRelation, Dict[str, Any]] = dict()
    assert len(document.metadata["relation_ids"]) == len(document.relations)
    for i, rel in enumerate(document.relations):
        arg1_id = span_dicts[rel.head]["id"]
        arg2_id = span_dicts[rel.tail]["id"]
        relation_dict = {
            "id": document.metadata["relation_ids"][i],
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

    annotation_dicts = {
        "spans": span_dicts,
        "relations": relation_dicts,
    }
    all_attribute_annotations = {
        "spans": document.span_attributes,
        "relations": document.relation_attributes,
    }
    attribute_dicts: Dict[Annotation, Dict[str, Any]] = dict()
    attribute_ids_per_target = defaultdict(list)
    for target_layer, attribute_id in document.metadata["attribute_ids"]:
        attribute_ids_per_target[target_layer].append(attribute_id)

    for target_layer, attribute_ids in attribute_ids_per_target.items():
        attribute_annotations = all_attribute_annotations[target_layer]
        assert len(attribute_ids) == len(attribute_annotations)
        for i, attribute_annotation in enumerate(attribute_annotations):
            target_id = annotation_dicts[target_layer][attribute_annotation.annotation]["id"]
            attribute_dict = {
                "id": attribute_ids_per_target[target_layer][i],
                "type": attribute_annotation.label,
                "target": target_id,
                "value": attribute_annotation.value,
            }
            if attribute_annotation in attribute_dicts:
                prev_ann_dict = attribute_dicts[attribute_annotation]
                ann_dict = attribute_annotation
                logger.warning(
                    f"document {document.id}: annotation exists twice: {prev_ann_dict['id']} and {ann_dict['id']} "
                    f"are identical"
                )
            attribute_dicts[attribute_annotation] = attribute_dict

    example["attributions"] = ld2dl(
        list(attribute_dicts.values()), keys=["id", "type", "target", "value"]
    )
    example["normalizations"] = ld2dl(
        [], keys=["id", "type", "target", "resource_id", "entity_id"]
    )
    example["notes"] = ld2dl([], keys=["id", "type", "target", "note"])

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
    BASE_DATASET_REVISION = "70446e79e089d5e5cd5f3426061991a2fcfbf529"

    def _generate_document(self, example, **kwargs):
        return example_to_document(
            example, merge_fragmented_spans=self.config.merge_fragmented_spans
        )
