import dataclasses

import pytest
from pytorch_ie import AnnotationLayer, annotation_field
from pytorch_ie.annotations import LabeledSpan
from pytorch_ie.documents import TextBasedDocument, TextDocumentWithLabeledSpans

from pie_datasets.document.processing import Caster, Converter, Pipeline


@dataclasses.dataclass
class ExampleDocument(TextBasedDocument):
    entities: AnnotationLayer[LabeledSpan] = annotation_field(target="text")


@pytest.fixture
def document():
    doc = ExampleDocument(text="Hello world!")
    doc.entities.append(LabeledSpan(start=0, end=5, label="greeting"))
    assert str(doc.entities[0]) == "Hello"
    doc.entities.append(LabeledSpan(start=6, end=11, label="location"))
    assert str(doc.entities[1]) == "world"
    return doc


def get_str_repr(obj):
    return obj.__module__ + "." + obj.__name__


@pytest.mark.parametrize(
    "document_type", [TextDocumentWithLabeledSpans, get_str_repr(TextDocumentWithLabeledSpans)]
)
def test_caster(document, document_type):
    caster = Caster(document_type=document_type, field_mapping={"entities": "labeled_spans"})
    casted_doc = caster(document)
    assert isinstance(casted_doc, TextDocumentWithLabeledSpans)
    assert len(casted_doc.labeled_spans) == len(document.entities)
    assert str(casted_doc.labeled_spans[0]) == "Hello"
    assert str(casted_doc.labeled_spans[1]) == "world"


def test_caster_resolve_wrong_type():
    with pytest.raises(TypeError) as excinfo:
        Caster(document_type=get_str_repr(test_caster_resolve_wrong_type))
    assert str(excinfo.value).startswith(
        "(resolved) document_type must be a subclass of Document, but it is: "
        "<function test_caster_resolve_wrong_type at"
    )


def add_suffix(document: ExampleDocument, suffix: str) -> ExampleDocument:
    result = ExampleDocument(text=document.text + suffix)
    for entity in document.entities:
        result.entities.append(entity.copy())
    return result


@pytest.mark.parametrize("function", [add_suffix, get_str_repr(add_suffix)])
def test_converter(document, function):
    converter = Converter(function=function, suffix="!!")
    converted_doc = converter(document)
    assert isinstance(converted_doc, ExampleDocument)
    assert converted_doc.text == document.text + "!!"
    assert len(converted_doc.entities) == len(document.entities)
    assert str(converted_doc.entities[0]) == "Hello"
    assert str(converted_doc.entities[1]) == "world"


def test_pipeline_dict(document):
    pipeline = Pipeline(
        converter=Converter(function=add_suffix, suffix="!!"),
        caster=Caster(
            document_type=TextDocumentWithLabeledSpans, field_mapping={"entities": "labeled_spans"}
        ),
    )
    converted_doc = pipeline(document)
    assert isinstance(converted_doc, TextDocumentWithLabeledSpans)
    assert converted_doc.text == document.text + "!!"
    assert len(converted_doc.labeled_spans) == len(document.entities)
    assert str(converted_doc.labeled_spans[0]) == "Hello"
    assert str(converted_doc.labeled_spans[1]) == "world"


def test_pipeline_list(document):
    pipeline = Pipeline(
        Converter(function=add_suffix, suffix="!!"),
        Caster(
            document_type=TextDocumentWithLabeledSpans, field_mapping={"entities": "labeled_spans"}
        ),
    )
    converted_doc = pipeline(document)
    assert isinstance(converted_doc, TextDocumentWithLabeledSpans)
    assert converted_doc.text == document.text + "!!"
    assert len(converted_doc.labeled_spans) == len(document.entities)
    assert str(converted_doc.labeled_spans[0]) == "Hello"
    assert str(converted_doc.labeled_spans[1]) == "world"


def test_pipeline_wrong_args():
    with pytest.raises(ValueError) as excinfo:
        Pipeline(
            Converter(function=add_suffix, suffix="!!"),
            caster=Caster(
                document_type=TextDocumentWithLabeledSpans,
                field_mapping={"entities": "labeled_spans"},
            ),
        )
    assert str(excinfo.value) == "You cannot use both positional and keyword arguments."
