from typing import Any, Union

import datasets
import pytest
from pie_modules.annotations import BinaryRelation, LabeledMultiSpan, LabeledSpan
from pytorch_ie.core import Annotation
from pytorch_ie.documents import TextBasedDocument

from dataset_builders.pie.brat.brat import Brat
from pie_datasets.builders.brat import (
    BratAttribute,
    BratDocument,
    BratDocumentWithMergedSpans,
    document_to_example,
    example_to_document,
)
from tests.dataset_builders.common import PIE_BASE_PATH, PIE_DS_FIXTURE_DATA_PATH

datasets.disable_caching()

DATASET_NAME = "brat"
PIE_DATASET_PATH = PIE_BASE_PATH / DATASET_NAME
HF_DATASET_PATH = Brat.BASE_DATASET_PATH
FIXTURE_DATA_PATH = PIE_DS_FIXTURE_DATA_PATH / DATASET_NAME
SPLIT_SIZES = {"train": 2}


def resolve_annotation(annotation: Annotation) -> Any:
    if annotation.target is None:
        return None
    if isinstance(annotation, LabeledMultiSpan):
        return (
            [annotation.target[start:end] for start, end in annotation.slices],
            annotation.label,
        )
    elif isinstance(annotation, LabeledSpan):
        return (annotation.target[annotation.start : annotation.end], annotation.label)
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
    else:
        raise TypeError(f"Unknown annotation type: {type(annotation)}")


@pytest.fixture(scope="module")
def hf_dataset():
    return datasets.load_dataset(str(HF_DATASET_PATH), data_dir=str(FIXTURE_DATA_PATH))


def test_hf_dataset(hf_dataset):
    assert set(hf_dataset) == set(SPLIT_SIZES)
    split_sizes = {split_name: len(ds) for split_name, ds in hf_dataset.items()}
    assert split_sizes == SPLIT_SIZES


@pytest.fixture(params=range(SPLIT_SIZES["train"]))
def sample_idx(request):
    return request.param


@pytest.fixture()
def hf_example(hf_dataset, sample_idx):
    return hf_dataset["train"][sample_idx]


def test_hf_example(hf_example, sample_idx):
    if sample_idx == 0:
        assert hf_example == {
            "context": "Jane lives in Berlin.\n",
            "file_name": "1",
            "spans": {
                "id": ["T1", "T2"],
                "type": ["person", "city"],
                "locations": [{"start": [0], "end": [4]}, {"start": [14], "end": [20]}],
                "text": ["Jane", "Berlin"],
            },
            "relations": {"id": [], "type": [], "arguments": []},
            "equivalence_relations": {"type": [], "targets": []},
            "events": {"id": [], "type": [], "trigger": [], "arguments": []},
            "attributions": {"id": [], "type": [], "target": [], "value": []},
            "normalizations": {
                "id": [],
                "type": [],
                "target": [],
                "resource_id": [],
                "entity_id": [],
            },
            "notes": {"id": [], "type": [], "target": [], "note": []},
        }
    elif sample_idx == 1:
        assert hf_example == {
            "context": "Seattle is a rainy city. Jenny Durkan is the city's mayor.\n",
            "file_name": "2",
            "spans": {
                "id": ["T1", "T2"],
                "type": ["city", "person"],
                "locations": [{"start": [0], "end": [7]}, {"start": [25], "end": [37]}],
                "text": ["Seattle", "Jenny Durkan"],
            },
            "relations": {
                "id": ["R1"],
                "type": ["mayor_of"],
                "arguments": [{"type": ["Arg1", "Arg2"], "target": ["T2", "T1"]}],
            },
            "equivalence_relations": {"type": [], "targets": []},
            "events": {"id": [], "type": [], "trigger": [], "arguments": []},
            "attributions": {
                "id": ["A1", "A2"],
                "type": ["factuality", "statement"],
                "target": ["T1", "R1"],
                "value": ["actual", "true"],
            },
            "normalizations": {
                "id": [],
                "type": [],
                "target": [],
                "resource_id": [],
                "entity_id": [],
            },
            "notes": {"id": [], "type": [], "target": [], "note": []},
        }
    else:
        raise ValueError(f"Unknown sample index: {sample_idx}")


@pytest.fixture(
    params=[config.name for config in Brat.BUILDER_CONFIGS],  # scope="module"
)
def pie_dataset_variant(request):
    return request.param


@pytest.fixture()
def generated_document(
    hf_example, hf_dataset, pie_dataset_variant
) -> Union[BratDocument, BratDocumentWithMergedSpans]:
    builder = Brat(name=pie_dataset_variant)
    kwargs = builder._generate_document_kwargs(hf_dataset["train"]) or {}
    document = builder._generate_document(example=hf_example, **kwargs)
    assert document is not None
    return document


def test_generate_document(generated_document, pie_dataset_variant, sample_idx):
    assert generated_document is not None
    resolved_spans = [resolve_annotation(annotation=span) for span in generated_document.spans]
    resolved_relations = [
        resolve_annotation(relation) for relation in generated_document.relations
    ]
    if sample_idx == 0:
        assert len(generated_document.spans) == 2
        assert len(generated_document.relations) == 0
        assert len(generated_document.span_attributes) == 0
        assert len(generated_document.relation_attributes) == 0

        if pie_dataset_variant == "default":
            assert resolved_spans[0] == (["Jane"], "person")
            assert resolved_spans[1] == (["Berlin"], "city")
        elif pie_dataset_variant == "merge_fragmented_spans":
            assert resolved_spans[0] == ("Jane", "person")
            assert resolved_spans[1] == ("Berlin", "city")
        else:
            raise ValueError(f"Unknown dataset variant: {pie_dataset_variant}")

    elif sample_idx == 1:
        assert len(generated_document.spans) == 2
        assert len(generated_document.relations) == 1
        assert len(generated_document.span_attributes) == 1
        assert len(generated_document.relation_attributes) == 1

        resolved_span_attributes = [
            resolve_annotation(attribute) for attribute in generated_document.span_attributes
        ]
        resolved_relation_attributes = [
            resolve_annotation(attribute) for attribute in generated_document.relation_attributes
        ]

        if pie_dataset_variant == "default":
            assert resolved_spans[0] == (["Seattle"], "city")
            assert resolved_spans[1] == (["Jenny Durkan"], "person")
            assert resolved_relations[0] == (
                (["Jenny Durkan"], "person"),
                "mayor_of",
                (["Seattle"], "city"),
            )
            assert resolved_span_attributes[0] == ((["Seattle"], "city"), "factuality", "actual")
            assert resolved_relation_attributes[0] == (
                ((["Jenny Durkan"], "person"), "mayor_of", (["Seattle"], "city")),
                "statement",
                "true",
            )
        elif pie_dataset_variant == "merge_fragmented_spans":
            assert resolved_spans[0] == ("Seattle", "city")
            assert resolved_spans[1] == ("Jenny Durkan", "person")
            assert resolved_relations[0] == (
                ("Jenny Durkan", "person"),
                "mayor_of",
                ("Seattle", "city"),
            )
            assert resolved_span_attributes[0] == (("Seattle", "city"), "factuality", "actual")
            assert resolved_relation_attributes[0] == (
                (("Jenny Durkan", "person"), "mayor_of", ("Seattle", "city")),
                "statement",
                "true",
            )
        else:
            raise ValueError(f"Unknown dataset variant: {pie_dataset_variant}")
    else:
        raise ValueError(f"Unknown sample index: {sample_idx}")


@pytest.mark.parametrize("merge_fragmented_spans", [True, False])
def test_example_to_document_and_back_all(hf_dataset, merge_fragmented_spans):
    for split_name, split in hf_dataset.items():
        for hf_example in split:
            doc = example_to_document(hf_example, merge_fragmented_spans=merge_fragmented_spans)
            assert isinstance(doc, TextBasedDocument)
            hf_example_back = document_to_example(doc)
            assert hf_example == hf_example_back
