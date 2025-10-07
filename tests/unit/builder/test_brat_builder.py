import dataclasses
import logging
from typing import Union

import pytest
from datasets import DatasetBuilder, disable_caching
from datasets.load import load_dataset_builder
from pie_core import AnnotationLayer, annotation_field
from pie_documents.annotations import LabeledSpan
from pie_documents.documents import TextBasedDocument

from pie_datasets.builders.brat import (
    BratAttribute,
    BratBuilder,
    BratDocument,
    BratDocumentWithMergedSpans,
    BratNote,
)

HF_EXAMPLES = [
    {
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
        "notes": {
            "id": ["#1"],
            "type": ["AnnotatorNotes"],
            "target": ["T1"],
            "note": ["last name is omitted"],
        },
    },
    {
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
        "notes": {
            "id": ["#1"],
            "type": ["AnnotatorNotes"],
            "target": ["R1"],
            "note": ["single relation"],
        },
    },
]

# TODO: check if this really makes errors more visible (or does this hide caching related errors)?
# disable HF caching
# disable_caching()


@pytest.fixture(scope="module")
def base_builder() -> DatasetBuilder:
    result = load_dataset_builder(path=BratBuilder.BASE_DATASET_PATH)
    return result


def test_base_builder(base_builder):
    assert base_builder is not None
    assert base_builder.name == "brat"
    assert base_builder.config.name == "default"


@pytest.fixture(scope="module", params=BratBuilder.BUILDER_CONFIGS)
def config_name(request) -> str:
    return request.param.name


def test_config_names(config_name):
    assert config_name in ["default", "merge_fragmented_spans"]


@pytest.fixture(scope="module")
def builder(config_name: str) -> BratBuilder:
    return BratBuilder(config_name=config_name)


def test_builder(builder):
    assert builder is not None


@pytest.fixture(scope="module", params=HF_EXAMPLES)
def hf_example(request) -> dict:
    return request.param


def test_generate_document(builder, hf_example):
    kwargs = dict()
    generated_document: Union[BratDocument, BratDocumentWithMergedSpans] = (
        builder._generate_document(example=hf_example, **kwargs)
    )

    if hf_example == HF_EXAMPLES[0]:
        assert len(generated_document.relations) == 0
        assert len(generated_document.attributes) == 0

        if builder.config.name == "default":
            assert generated_document.spans.resolve() == [
                ("person", ("Jane",)),
                ("city", ("Berlin",)),
            ]
            assert generated_document.notes.resolve() == [
                ("last name is omitted", "AnnotatorNotes", ("person", ("Jane",)))
            ]
        elif builder.config.name == "merge_fragmented_spans":
            assert generated_document.spans.resolve() == [("person", "Jane"), ("city", "Berlin")]
            assert generated_document.notes.resolve() == [
                ("last name is omitted", "AnnotatorNotes", ("person", "Jane"))
            ]
        else:
            raise ValueError(f"Unknown builder variant: {builder.name}")

    elif hf_example == HF_EXAMPLES[1]:
        if builder.config.name == "default":
            assert generated_document.spans.resolve() == [
                ("city", ("Seattle",)),
                ("person", ("Jenny Durkan",)),
            ]
            assert generated_document.relations.resolve() == [
                ("mayor_of", (("person", ("Jenny Durkan",)), ("city", ("Seattle",))))
            ]
            assert generated_document.attributes.resolve() == [
                ("actual", "factuality", ("city", ("Seattle",))),
                (
                    "true",
                    "statement",
                    ("mayor_of", (("person", ("Jenny Durkan",)), ("city", ("Seattle",)))),
                ),
            ]

            assert generated_document.notes.resolve() == [
                (
                    "single relation",
                    "AnnotatorNotes",
                    ("mayor_of", (("person", ("Jenny Durkan",)), ("city", ("Seattle",)))),
                )
            ]
        elif builder.config.name == "merge_fragmented_spans":
            assert generated_document.spans.resolve() == [
                ("city", "Seattle"),
                ("person", "Jenny Durkan"),
            ]
            assert generated_document.relations.resolve() == [
                ("mayor_of", (("person", "Jenny Durkan"), ("city", "Seattle")))
            ]
            assert generated_document.attributes.resolve() == [
                ("actual", "factuality", ("city", "Seattle")),
                (
                    "true",
                    "statement",
                    ("mayor_of", (("person", "Jenny Durkan"), ("city", "Seattle"))),
                ),
            ]
            assert generated_document.notes.resolve() == [
                (
                    "single relation",
                    "AnnotatorNotes",
                    ("mayor_of", (("person", "Jenny Durkan"), ("city", "Seattle"))),
                )
            ]
        else:
            raise ValueError(f"Unknown builder variant: {config_name}")
    else:
        raise ValueError(f"Unknown sample: {hf_example}")


def test_example_to_document_and_back_all(builder):
    for hf_example in HF_EXAMPLES:
        doc = builder._generate_document(hf_example)
        assert isinstance(doc, builder.document_type)
        hf_example_back = builder._generate_example(doc)
        assert hf_example == hf_example_back


def test_document_to_example_wrong_type(builder):
    doc = TextBasedDocument(text="Hello, world!")

    with pytest.raises(TypeError) as exc_info:
        builder._generate_example(doc)
    assert str(exc_info.value) == f"document type {type(doc)} is not supported"


def test_example_to_document_missing_attribute_target(builder):
    example = HF_EXAMPLES[1].copy()
    example["attributions"] = {
        "id": ["A1"],
        "type": ["factuality"],
        "target": [
            "N1",
        ],
        "value": ["actual"],
    }
    with pytest.raises(Exception) as exc_info:
        builder._generate_document(example=example)
    assert (
        str(exc_info.value)
        == "attribute target annotation N1 not found in any of the target layers (spans, relations)"
    )


def test_example_to_document_missing_note_target(builder):
    example = HF_EXAMPLES[0].copy()
    example["notes"] = {
        "id": ["#1"],
        "type": ["AnnotatorNotes"],
        "target": ["T3"],
        "note": ["last name is omitted"],
    }

    with pytest.raises(Exception) as exc_info:
        builder._generate_document(example=example)
    assert (
        str(exc_info.value)
        == "note target annotation T3 not found in any of the target layers (spans, relations, attributes)"
    )


def test_document_to_example_warnings(builder, caplog):
    example = HF_EXAMPLES[0].copy()
    example["notes"] = {
        "id": ["#1", "#2"],
        "type": ["AnnotatorNotes", "AnnotatorNotes"],
        "target": ["T1", "T1"],
        "note": ["last name is omitted", "last name is omitted"],
    }

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        doc = builder._generate_document(example)
    assert caplog.messages == ["document 1: annotation exists twice: #1 and #2 are identical"]

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        builder._generate_example(doc)
    assert caplog.messages == ["document 1: annotation exists twice: #1 and #2 are identical"]

    example = HF_EXAMPLES[1].copy()
    example["attributions"] = {
        "id": ["A1", "A2"],
        "type": ["factuality", "factuality"],
        "target": ["T1", "T1"],
        "value": ["actual", "actual"],
    }

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        doc = builder._generate_document(example)
    assert caplog.messages == ["document 2: annotation exists twice: A1 and A2 are identical"]

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        builder._generate_example(doc)
    assert caplog.messages == ["document 2: annotation exists twice: A1 and A2 are identical"]


def test_brat_attribute():
    @dataclasses.dataclass
    class ExampleDocument(TextBasedDocument):
        spans: AnnotationLayer[LabeledSpan] = annotation_field(target="text")
        attributes: AnnotationLayer[BratAttribute] = annotation_field(target="spans")

    doc = ExampleDocument(text="Jane lives in Berlin.")
    span = LabeledSpan(start=0, end=4, label="person")
    doc.spans.append(span)

    span_attribute = BratAttribute(annotation=span, label="actual")
    doc.attributes.append(span_attribute)

    assert span_attribute.resolve() == (True, "actual", ("person", "Jane"))

    attribute_with_value = BratAttribute(annotation=span, label="actual", value="maybe")
    doc.attributes.append(attribute_with_value)
    assert attribute_with_value.resolve() == ("maybe", "actual", ("person", "Jane"))


def test_brat_note():
    @dataclasses.dataclass
    class ExampleDocument(TextBasedDocument):
        spans: AnnotationLayer[LabeledSpan] = annotation_field(target="text")
        notes: AnnotationLayer[BratNote] = annotation_field(target="spans")

    doc = ExampleDocument(text="Jane lives in Berlin.")
    span = LabeledSpan(start=0, end=4, label="person")
    doc.spans.append(span)

    note = BratNote(annotation=span, label="AnnotatorNotes", value="not sure if this is correct")
    doc.notes.append(note)

    assert note.resolve() == ("not sure if this is correct", "AnnotatorNotes", ("person", "Jane"))
