import dataclasses
import json
import logging
import os
import re
from pathlib import Path
from typing import List, Optional

from pytorch_ie.annotations import BinaryRelation, LabeledSpan
from pytorch_ie.core import AnnotationList, annotation_field
from pytorch_ie.documents import TokenBasedDocument

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
