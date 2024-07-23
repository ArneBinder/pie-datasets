from typing import Any

import pytest
from pytorch_ie.annotations import BinaryRelation, LabeledMultiSpan, LabeledSpan
from pytorch_ie.core import Annotation
from pytorch_ie.documents import TextBasedDocument

from src.pie_datasets.builders.brat import BratAttribute, BratBuilder, BratNote

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
    elif isinstance(annotation, BratNote):
        result = (resolve_annotation(annotation.annotation), annotation.label)
        if annotation.value is not None:
            return result + (annotation.value,)
        else:
            return result
    else:
        raise TypeError(f"Unknown annotation type: {type(annotation)}")


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
    generated_document = builder._generate_document(example=hf_example, **kwargs)
    resolved_spans = [resolve_annotation(annotation=span) for span in generated_document.spans]
    resolved_relations = [
        resolve_annotation(relation) for relation in generated_document.relations
    ]
    if hf_example == HF_EXAMPLES[0]:
        assert len(generated_document.spans) == 2
        assert len(generated_document.relations) == 0
        assert len(generated_document.span_attributes) == 0
        assert len(generated_document.relation_attributes) == 0
        assert len(generated_document.span_notes) == 1
        assert len(generated_document.relation_notes) == 0

        resolved_span_notes = [resolve_annotation(note) for note in generated_document.span_notes]

        if builder.config.name == "default":
            assert resolved_spans[0] == (["Jane"], "person")
            assert resolved_spans[1] == (["Berlin"], "city")
            assert resolved_span_notes[0] == (
                (["Jane"], "person"),
                "AnnotatorNotes",
                "last name is omitted",
            )
        elif builder.config.name == "merge_fragmented_spans":
            assert resolved_spans[0] == ("Jane", "person")
            assert resolved_spans[1] == ("Berlin", "city")
            assert resolved_span_notes[0] == (
                ("Jane", "person"),
                "AnnotatorNotes",
                "last name is omitted",
            )
        else:
            raise ValueError(f"Unknown builder variant: {builder.name}")

    elif hf_example == HF_EXAMPLES[1]:
        assert len(generated_document.spans) == 2
        assert len(generated_document.relations) == 1
        assert len(generated_document.span_attributes) == 1
        assert len(generated_document.relation_attributes) == 1
        assert len(generated_document.span_notes) == 0
        assert len(generated_document.relation_notes) == 1

        resolved_span_attributes = [
            resolve_annotation(attribute) for attribute in generated_document.span_attributes
        ]
        resolved_relation_attributes = [
            resolve_annotation(attribute) for attribute in generated_document.relation_attributes
        ]
        resolved_relation_notes = [
            resolve_annotation(note) for note in generated_document.relation_notes
        ]

        if builder.config.name == "default":
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
            assert resolved_relation_notes[0] == (
                ((["Jenny Durkan"], "person"), "mayor_of", (["Seattle"], "city")),
                "AnnotatorNotes",
                "single relation",
            )
        elif builder.config.name == "merge_fragmented_spans":
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
            assert resolved_relation_notes[0] == (
                (("Jenny Durkan", "person"), "mayor_of", ("Seattle", "city")),
                "AnnotatorNotes",
                "single relation",
            )
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
