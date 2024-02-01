import dataclasses
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, List, Optional, Sequence

from pytorch_ie import Annotation
from pytorch_ie.annotations import BinaryRelation, LabeledMultiSpan, LabeledSpan, Span
from pytorch_ie.core import AnnotationList, annotation_field
from pytorch_ie.documents import TokenBasedDocument

from pie_datasets.builders.brat import BratAttribute
from tests import FIXTURES_ROOT

DATASET_BUILDER_BASE_PATH = Path("dataset_builders")
HF_BASE_PATH = DATASET_BUILDER_BASE_PATH / "hf"
PIE_BASE_PATH = DATASET_BUILDER_BASE_PATH / "pie"
HF_DS_FIXTURE_DATA_PATH = FIXTURES_ROOT / "dataset_builders" / "hf"
PIE_DS_FIXTURE_DATA_PATH = FIXTURES_ROOT / "dataset_builders" / "pie"

logger = logging.getLogger(__name__)


def _deep_compare(
    obj,
    obj_expected,
    path: Optional[str] = None,
    excluded_paths: Optional[List[str]] = None,
    enforce_equal_dict_keys: bool = True,
):
    if path is not None and excluded_paths is not None:
        for excluded_path in excluded_paths:
            if re.match(excluded_path, path):
                return

    if type(obj) != type(obj_expected):
        raise AssertionError(f"{path}: {obj} != {obj_expected}")
    if isinstance(obj, (list, tuple)):
        if len(obj) != len(obj_expected):
            raise AssertionError(f"{path}: {obj} != {obj_expected}")
        for i in range(len(obj)):
            _deep_compare(
                obj[i],
                obj_expected[i],
                path=f"{path}.{i}" if path is not None else str(i),
                excluded_paths=excluded_paths,
                enforce_equal_dict_keys=enforce_equal_dict_keys,
            )
    elif isinstance(obj, dict):
        if enforce_equal_dict_keys and obj.keys() != obj_expected.keys():
            raise AssertionError(f"{path}: {obj} != {obj_expected}")
        for k in set(obj) | set(obj_expected):
            _deep_compare(
                obj.get(k, None),
                obj_expected.get(k, None),
                path=f"{path}.{k}" if path is not None else str(k),
                excluded_paths=excluded_paths,
                enforce_equal_dict_keys=enforce_equal_dict_keys,
            )
    else:
        if obj != obj_expected:
            raise AssertionError(f"{path}: {obj} != {obj_expected}")


def _dump_json(obj, fn):
    logger.warning(f"dump fixture data: {fn}")
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    with open(fn, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _load_json(fn: str):
    with open(fn) as f:
        ex = json.load(f)
    return ex


def resolve_annotation(annotation: Annotation) -> Any:
    if annotation.target is None:
        return None
    if isinstance(annotation, LabeledMultiSpan):
        return (
            [annotation.target[start:end] for start, end in annotation.slices],
            annotation.label,
        )
    elif isinstance(annotation, LabeledSpan):
        return annotation.target[annotation.start : annotation.end], annotation.label
    elif isinstance(annotation, BinaryRelation):
        return (
            resolve_annotation(annotation.head),
            annotation.label,
            resolve_annotation(annotation.tail),
        )
    elif isinstance(annotation, BratAttribute):
        result = (resolve_annotation(annotation.annotation), annotation.label)
        if annotation.value is not None:
            return result + (annotation.value,)
        else:
            return result
    elif isinstance(annotation, BinaryRelation):
        return (
            resolve_annotation(annotation.head),
            annotation.label,
            resolve_annotation(annotation.tail),
        )
    else:
        raise TypeError(f"Unknown annotation type: {type(annotation)}")


def sort_annotations(annotations: Sequence[Annotation]) -> List[Annotation]:
    if len(annotations) == 0:
        return []
    annotation = annotations[0]
    if isinstance(annotation, LabeledSpan):
        return sorted(annotations, key=lambda a: (a.start, a.end, a.label))
    elif isinstance(annotation, Span):
        return sorted(annotations, key=lambda a: (a.start, a.end))
    elif isinstance(annotation, BinaryRelation):
        if isinstance(annotation.head, LabeledSpan) and isinstance(annotation.tail, LabeledSpan):
            return sorted(
                annotations,
                key=lambda a: (a.head.start, a.head.end, a.label, a.tail.start, a.tail.end),
            )
        else:
            raise ValueError(
                f"Unsupported relation type for BinaryRelation arguments: "
                f"{type(annotation.head)}, {type(annotation.tail)}"
            )
    else:
        raise ValueError(f"Unsupported annotation type: {type(annotation)}")


def resolve_annotations(annotations: Sequence[Annotation]) -> List[Any]:
    sorted_annotations = sort_annotations(annotations)
    return [resolve_annotation(a) for a in sorted_annotations]


@dataclasses.dataclass
class TestTokenDocumentWithLabeledSpans(TokenBasedDocument):
    labeled_spans: AnnotationList[LabeledSpan] = annotation_field(target="tokens")


@dataclasses.dataclass
class TestTokenDocumentWithLabeledSpansAndBinaryRelations(TestTokenDocumentWithLabeledSpans):
    binary_relations: AnnotationList[BinaryRelation] = annotation_field(target="labeled_spans")


@dataclasses.dataclass
class TestTokenDocumentWithLabeledPartitions(TokenBasedDocument):
    labeled_partitions: AnnotationList[LabeledSpan] = annotation_field(target="tokens")


@dataclasses.dataclass
class TestTokenDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions(
    TestTokenDocumentWithLabeledSpansAndBinaryRelations, TestTokenDocumentWithLabeledPartitions
):
    pass
