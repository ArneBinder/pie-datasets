from typing import Union

import pytest
from pytorch_ie.documents import TextBasedDocument

from pie_datasets.builders.brat import (
    BratBuilder,
    BratDocument,
    BratDocumentWithMergedSpans,
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


@pytest.fixture(scope="module", params=BratBuilder.BUILDER_CONFIGS)
def config_name(request) -> str:
    return request.param.name


def test_config_names(config_name):
    assert config_name in ["default", "merge_fragmented_spans"]


@pytest.fixture(scope="module")
def builder(config_name: str) -> BratBuilder:
    return BratBuilder(name=config_name)


def test_builder(builder):
    assert builder is not None


@pytest.fixture(scope="module", params=HF_EXAMPLES)
def hf_example(request) -> dict:
    return request.param


def test_generate_document(builder, hf_example):
    kwargs = dict()
    generated_document: Union[
        BratDocument, BratDocumentWithMergedSpans
    ] = builder._generate_document(example=hf_example, **kwargs)

    if hf_example == HF_EXAMPLES[0]:
        assert len(generated_document.relations) == 0
        assert len(generated_document.span_attributes) == 0
        assert len(generated_document.relation_attributes) == 0

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
            assert generated_document.span_attributes.resolve() == [
                ("actual", "factuality", ("city", ("Seattle",)))
            ]
            assert generated_document.relation_attributes.resolve() == [
                (
                    "true",
                    "statement",
                    ("mayor_of", (("person", ("Jenny Durkan",)), ("city", ("Seattle",)))),
                )
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
            assert generated_document.span_attributes.resolve() == [
                ("actual", "factuality", ("city", "Seattle"))
            ]
            assert generated_document.relation_attributes.resolve() == [
                (
                    "true",
                    "statement",
                    ("mayor_of", (("person", "Jenny Durkan"), ("city", "Seattle"))),
                )
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
