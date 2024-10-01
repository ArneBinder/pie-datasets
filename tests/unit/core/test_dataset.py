from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy
import pytest
import torch
from pyexpat import features
from pytorch_ie import Document
from pytorch_ie.annotations import BinaryRelation, Label, LabeledSpan, Span
from pytorch_ie.core import AnnotationList, annotation_field
from pytorch_ie.core.taskmodule import (
    IterableTaskEncodingDataset,
    TaskEncodingDataset,
    TaskEncodingSequence,
)
from pytorch_ie.documents import TextBasedDocument
from pytorch_ie.taskmodules import TransformerSpanClassificationTaskModule

from pie_datasets import Dataset, IterableDataset
from pie_datasets.core.dataset import (
    _add_dset_name_to_document,
    concatenate_datasets,
    get_pie_dataset_type,
)
from tests.conftest import TestDocument
from tests.unit.core import TEST_PACKAGE

TEST_MODULE = f"{TEST_PACKAGE}.{Path(__file__).stem}"


def test_dataset(maybe_iterable_dataset):
    dataset = {
        k: list(v) if isinstance(v, IterableDataset) else v
        for k, v in maybe_iterable_dataset.items()
    }
    assert set(dataset.keys()) == {"train", "validation", "test"}

    assert len(dataset["train"]) == 8
    assert len(dataset["validation"]) == 2
    assert len(dataset["test"]) == 2

    train_doc5 = dataset["train"][4]
    assert train_doc5.id == "train_doc5"
    assert len(train_doc5.sentences) == 3
    assert len(train_doc5.entities) == 3
    assert len(train_doc5.relations) == 3

    assert str(train_doc5.sentences[1]) == "Entity G works at H."


def test_dataset_index(dataset):
    train_dataset = dataset["train"]
    assert train_dataset[4].id == "train_doc5"
    assert [doc.id for doc in train_dataset[0, 3, 5]] == ["train_doc1", "train_doc4", "train_doc6"]
    assert [doc.id for doc in train_dataset[2:5]] == ["train_doc3", "train_doc4", "train_doc5"]


def test_dataset_map(maybe_iterable_dataset):
    train_dataset = maybe_iterable_dataset["train"]

    def clear_relations(document):
        document.relations.clear()
        return document

    assert sum(len(doc.relations) for doc in train_dataset) == 7

    mapped_dataset1 = train_dataset.map(clear_relations)

    assert sum(len(doc.relations) for doc in mapped_dataset1) == 0
    assert sum(len(doc.relations) for doc in train_dataset) == 7


def test_dataset_map_batched(maybe_iterable_dataset):
    train_dataset = maybe_iterable_dataset["train"]

    def clear_relations_batched(documents):
        assert len(documents) == 2
        for document in documents:
            document.relations.clear()
        return documents

    assert sum(len(doc.relations) for doc in train_dataset) == 7

    mapped_dataset1 = train_dataset.map(clear_relations_batched, batched=True, batch_size=2)

    assert sum(len(doc.relations) for doc in mapped_dataset1) == 0
    assert sum(len(doc.relations) for doc in train_dataset) == 7


def test_dataset_map_with_result_document_type(maybe_iterable_dataset):
    @dataclass
    class TestDocument(TextBasedDocument):
        sentences: AnnotationList[Span] = annotation_field(target="text")
        entities: AnnotationList[LabeledSpan] = annotation_field(target="text")
        relations: AnnotationList[BinaryRelation] = annotation_field(target="entities")

    @dataclass
    class TestDocumentWithTokensButNoRelations(TextBasedDocument):
        sentences: AnnotationList[Span] = annotation_field(target="text")
        tokens: AnnotationList[Span] = annotation_field(target="text")
        entities: AnnotationList[LabeledSpan] = annotation_field(target="text")

    def clear_relations_and_add_one_token(
        document: TestDocument,
    ) -> TestDocumentWithTokensButNoRelations:
        document.relations.clear()
        # the conversion here is not really necessary, but to have correct typing
        result = document.as_type(TestDocumentWithTokensButNoRelations)
        # subtract 1 to create a Span different from the sentence to account for
        # https://github.com/ChristophAlt/pytorch-ie/pull/222
        result.tokens.append(Span(0, len(document.text) - 1))
        return result

    train_dataset = maybe_iterable_dataset["train"]

    assert sum(len(doc.relations) for doc in train_dataset) == 7

    mapped_dataset1 = train_dataset.map(
        clear_relations_and_add_one_token,
        result_document_type=TestDocumentWithTokensButNoRelations,
    )

    assert sum(len(doc.relations) for doc in train_dataset) == 7

    doc0 = list(train_dataset)[0]
    doc0_mapped = list(mapped_dataset1)[0]
    assert len(doc0_mapped.tokens) == 1
    token = doc0_mapped.tokens[0]
    assert token.start == 0
    assert token.end == len(doc0.text) - 1
    # check field names because isinstance does not work (the code of the document types
    # is the same, but lives at different locations)
    assert {f.name for f in doc0.fields()} == {f.name for f in TestDocument.fields()}
    assert {f.name for f in doc0_mapped.fields()} == {
        f.name for f in TestDocumentWithTokensButNoRelations.fields()
    }


def test_get_pie_dataset_type(hf_dataset, iterable_hf_dataset):
    assert get_pie_dataset_type(hf_dataset["train"]) == Dataset
    assert get_pie_dataset_type(iterable_hf_dataset["train"]) == IterableDataset
    with pytest.raises(TypeError) as excinfo:
        get_pie_dataset_type("not a dataset")
    assert (
        str(excinfo.value)
        == "the dataset must be of type Dataset or IterableDataset, but is of type <class 'str'>"
    )


@dataclass
class TestDocumentWithLabel(TextBasedDocument):
    label: AnnotationList[Label] = annotation_field()


def convert_to_document_with_label(document: TestDocument) -> TestDocumentWithLabel:
    result = TestDocumentWithLabel(text=document.text)
    result.label.append(Label(label="label"))
    return result


@pytest.fixture
def dataset_with_converter_functions(maybe_iterable_dataset) -> Union[Dataset, IterableDataset]:
    train_dataset: Union[Dataset, IterableDataset] = maybe_iterable_dataset["train"]
    assert len(train_dataset.document_converters) == 0

    train_dataset.register_document_converter(convert_to_document_with_label)
    return train_dataset


def test_register_document_converter_function(dataset_with_converter_functions):
    assert len(dataset_with_converter_functions.document_converters) == 1
    assert TestDocumentWithLabel in dataset_with_converter_functions.document_converters
    assert (
        dataset_with_converter_functions.document_converters[TestDocumentWithLabel]
        == convert_to_document_with_label
    )


@dataclass
class TestDocumentWithLabeledSpans(TextBasedDocument):
    spans: AnnotationList[LabeledSpan] = annotation_field(target="text")


@pytest.fixture
def dataset_with_converter_mapping(maybe_iterable_dataset) -> Union[Dataset, IterableDataset]:
    train_dataset: Union[Dataset, IterableDataset] = maybe_iterable_dataset["train"]
    assert len(train_dataset.document_converters) == 0

    field_mapping = {"entities": "spans"}
    train_dataset.register_document_converter(
        converter=field_mapping, document_type=TestDocumentWithLabeledSpans
    )
    return train_dataset


def test_register_document_converter_mapping(dataset_with_converter_mapping):
    assert len(dataset_with_converter_mapping.document_converters) == 1
    assert TestDocumentWithLabeledSpans in dataset_with_converter_mapping.document_converters
    assert dataset_with_converter_mapping.document_converters[TestDocumentWithLabeledSpans] == {
        "entities": "spans"
    }


def test_to_document_type_function(dataset_with_converter_functions):
    # Features are only available for Dataset type (not for IterableDataset)
    if isinstance(dataset_with_converter_functions, Dataset):
        assert set(dataset_with_converter_functions.features) == {
            "entities",
            "relations",
            "metadata",
            "sentences",
            "id",
            "text",
        }
    else:
        assert dataset_with_converter_functions.features is None
    assert dataset_with_converter_functions.document_type == TestDocument
    converted_dataset = dataset_with_converter_functions.to_document_type(TestDocumentWithLabel)
    assert converted_dataset.document_type == TestDocumentWithLabel
    if isinstance(converted_dataset, Dataset):
        assert set(converted_dataset.features) == {"id", "label", "metadata", "text"}
    else:
        assert converted_dataset.features is None

    assert len(converted_dataset.document_converters) == 0
    for doc in converted_dataset:
        assert isinstance(doc, TestDocumentWithLabel)
        assert len(doc.label) == 1
        assert doc.label[0].label == "label"


def test_to_document_type_mapping(dataset_with_converter_mapping):
    assert dataset_with_converter_mapping.document_type == TestDocument
    converted_dataset = dataset_with_converter_mapping.to_document_type(
        TestDocumentWithLabeledSpans
    )
    assert converted_dataset.document_type == TestDocumentWithLabeledSpans

    assert len(converted_dataset.document_converters) == 0
    for doc_converted, doc in zip(converted_dataset, dataset_with_converter_mapping):
        assert isinstance(doc, TestDocument)
        assert isinstance(doc_converted, TestDocumentWithLabeledSpans)
        assert "spans" in doc_converted
        assert doc_converted.spans == doc.entities
        original_annotation_field_names = {f.name for f in doc.annotation_fields()}
        assert original_annotation_field_names == {"sentences", "entities", "relations"}
        for annotation_field_name in original_annotation_field_names:
            assert annotation_field_name not in doc_converted


def test_to_document_type_noop(maybe_iterable_dataset):
    train_dataset: Union[Dataset, IterableDataset] = maybe_iterable_dataset["train"]
    assert len(train_dataset.document_converters) == 0
    train_dataset.register_document_converter(
        convert_to_document_with_label, document_type=TestDocument
    )
    assert train_dataset.document_type == TestDocument
    converted_dataset = train_dataset.to_document_type(TestDocument)
    # the conversion should be a noop
    assert converted_dataset.document_type == TestDocument
    assert converted_dataset == train_dataset
    assert len(converted_dataset.document_converters) == 1
    assert TestDocument in converted_dataset.document_converters
    assert converted_dataset.document_converters[TestDocument] == convert_to_document_with_label


def test_to_document_type_convert_and_cast(dataset_with_converter_functions):
    @dataclass
    class TestDocumentWithLabelAndSpans(TestDocumentWithLabel):
        label: AnnotationList[Label] = annotation_field()
        spans: AnnotationList[Span] = annotation_field(target="text")

    assert dataset_with_converter_functions.document_type == TestDocument
    # The only converter is registered for TestDocumentWithLabel, but we request a conversion to
    # TestDocumentWithLabelAndSpans which is a *subclass* of TestDocumentWithLabel. This is a valid type
    # and the conversion is performed by first converting to TestDocumentWithLabel and then casting
    # to TestDocumentWithLabelAndSpans.
    converted_dataset = dataset_with_converter_functions.to_document_type(
        TestDocumentWithLabelAndSpans
    )
    assert converted_dataset.document_type == TestDocumentWithLabelAndSpans

    assert len(converted_dataset.document_converters) == 0
    for converted_doc, doc in zip(converted_dataset, dataset_with_converter_functions):
        assert isinstance(doc, TestDocument)
        assert isinstance(converted_doc, TestDocumentWithLabelAndSpans)
        assert converted_doc.text == doc.text
        assert len(converted_doc.label) == 1
        assert converted_doc.label[0].label == "label"
        assert len(converted_doc.spans) == 0


def test_to_document_type_not_found(dataset_with_converter_functions):
    assert dataset_with_converter_functions.document_type == TestDocument

    @dataclass
    class TestDocumentWithSpans(TestDocument):
        spans: AnnotationList[Span] = annotation_field(target="text")

    # The only converter is registered for TestDocumentWithLabel, but we request a conversion to
    # TestDocumentWithSpans. This is not a valid type because it is neither a subclass nor a superclass of
    # TestDocumentWithLabel, so an error is raised.
    with pytest.raises(ValueError) as excinfo:
        dataset_with_converter_functions.to_document_type(TestDocumentWithSpans)
    assert (
        str(excinfo.value)
        == "No valid key (either subclass or superclass) was found for the document type "
        f"'<class '{TEST_MODULE}.test_to_document_type_not_found.<locals>.TestDocumentWithSpans'>' "
        "in the document_converters of the dataset. Available keys: "
        "{<class '"
        + TEST_MODULE
        + ".TestDocumentWithLabel'>}. Consider adding a respective converter "
        "to the dataset with dataset.register_document_converter(my_converter_method) where "
        "my_converter_method should accept <class 'tests.conftest.TestDocument'> as input and return "
        f"'<class '{TEST_MODULE}.test_to_document_type_not_found.<locals>.TestDocumentWithSpans'>'."
    )


@pytest.fixture(scope="module")
def taskmodule():
    # TODO: use a mock taskmodule instead
    tokenizer_name_or_path = "bert-base-cased"
    taskmodule = TransformerSpanClassificationTaskModule(
        tokenizer_name_or_path=tokenizer_name_or_path,
        entity_annotation="entities",
    )
    return taskmodule


@pytest.fixture
def model_output():
    return {
        "logits": torch.from_numpy(
            numpy.log(
                [
                    # O, ORG, PER
                    [0.5, 0.2, 0.3],
                    [0.1, 0.1, 0.8],
                    [0.1, 0.5, 0.4],
                    [0.1, 0.4, 0.5],
                    [0.1, 0.6, 0.3],
                ]
            )
        ),
        "start_indices": torch.tensor([1, 1, 7, 1, 6]),
        "end_indices": torch.tensor([2, 4, 7, 4, 6]),
        "batch_indices": torch.tensor([0, 1, 1, 2, 2]),
    }


@pytest.mark.parametrize("encode_target", [False, True])
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("as_dataset", [False, True])
def test_dataset_with_taskmodule(
    maybe_iterable_dataset, taskmodule, model_output, encode_target, inplace, as_dataset
):
    train_dataset = maybe_iterable_dataset["train"]

    taskmodule.prepare(train_dataset)
    assert set(taskmodule.label_to_id.keys()) == {"PER", "ORG", "O"}
    assert [taskmodule.id_to_label[i] for i in range(3)] == ["O", "ORG", "PER"]
    assert taskmodule.label_to_id["O"] == 0

    as_task_encoding_sequence = not encode_target
    as_iterator = isinstance(train_dataset, (IterableDataset, Iterator))
    if as_task_encoding_sequence:
        if as_iterator:
            with pytest.raises(
                ValueError, match="can not return a TaskEncodingSequence as Iterator"
            ):
                taskmodule.encode(
                    train_dataset, encode_target=encode_target, as_dataset=as_dataset
                )
            return
        if as_dataset:
            with pytest.raises(
                ValueError, match="can not return a TaskEncodingSequence as a dataset"
            ):
                taskmodule.encode(
                    train_dataset, encode_target=encode_target, as_dataset=as_dataset
                )
            return

    task_encodings = taskmodule.encode(
        train_dataset, encode_target=encode_target, as_dataset=as_dataset
    )

    if as_iterator:
        if as_task_encoding_sequence:
            raise NotImplementedError("this is not yet implemented")
        if as_dataset:
            assert isinstance(task_encodings, IterableTaskEncodingDataset)
        else:
            assert isinstance(task_encodings, Iterator)
    else:
        if as_dataset:
            if as_task_encoding_sequence:
                raise NotImplementedError("this is not yet implemented")
            else:
                assert isinstance(task_encodings, TaskEncodingDataset)
        else:
            if as_task_encoding_sequence:
                assert isinstance(task_encodings, TaskEncodingSequence)
            else:
                assert isinstance(task_encodings, Sequence)

    task_encoding_list = list(task_encodings)
    assert len(task_encoding_list) == 8
    task_encoding = task_encoding_list[5]
    document = list(train_dataset)[5]
    assert task_encoding.document == document
    assert "input_ids" in task_encoding.inputs
    assert (
        taskmodule.tokenizer.decode(task_encoding.inputs["input_ids"], skip_special_tokens=True)
        == document.text
    )

    if encode_target:
        assert task_encoding.targets == [
            (1, 4, taskmodule.label_to_id["PER"]),
            (6, 6, taskmodule.label_to_id["ORG"]),
            (9, 9, taskmodule.label_to_id["ORG"]),
        ]
    else:
        assert not task_encoding.has_targets

    unbatched_outputs = taskmodule.unbatch_output(model_output)

    decoded_documents = taskmodule.decode(
        task_encodings=task_encodings,
        task_outputs=unbatched_outputs,
        inplace=inplace,
    )

    if isinstance(train_dataset, Dataset):
        assert len(decoded_documents) == len(train_dataset)

    assert {id(doc) for doc in decoded_documents}.isdisjoint({id(doc) for doc in train_dataset})

    expected_scores = [0.8, 0.5, 0.5, 0.6]
    i = 0
    for document in decoded_documents:
        for entity_expected, entity_decoded in zip(
            document["entities"], document["entities"].predictions
        ):
            assert entity_expected.start == entity_decoded.start
            assert entity_expected.end == entity_decoded.end
            assert entity_expected.label == entity_decoded.label
            assert expected_scores[i] == pytest.approx(entity_decoded.score)
            i += 1

    for document in train_dataset:
        assert not document["entities"].predictions


@pytest.mark.parametrize("as_iterable_dataset", [False, True])
def test_pie_dataset_from_documents(documents, as_iterable_dataset):
    if as_iterable_dataset:
        dataset_class = IterableDataset

        # make generators from list
        def _documents():
            yield from documents

        def _empty_docs():
            return iter([])

    else:
        dataset_class = Dataset
        _documents = documents
        _empty_docs = list[Document]()

    dataset_from_documents = dataset_class.from_documents(_documents)

    assert isinstance(dataset_from_documents, dataset_class)

    assert all(isinstance(doc, TextBasedDocument) for doc in dataset_from_documents)
    assert all(
        doc1.asdict() == doc2.asdict() for doc1, doc2 in zip(documents, dataset_from_documents)
    )
    assert hasattr(dataset_from_documents, "document_type")

    # Test dataset creation with document converter
    dataset_from_documents_with_converter = dataset_class.from_documents(
        _documents, document_converters={TestDocumentWithLabel: convert_to_document_with_label}
    )

    assert isinstance(dataset_from_documents_with_converter, dataset_class)

    assert len(dataset_from_documents_with_converter.document_converters) == 1
    assert TestDocumentWithLabel in dataset_from_documents_with_converter.document_converters
    assert (
        dataset_from_documents_with_converter.document_converters[TestDocumentWithLabel]
        == convert_to_document_with_label
    )

    # Test dataset creation with empty list / generator
    with pytest.raises(ValueError) as excinfo:
        dataset_class.from_documents(_empty_docs)
    assert str(excinfo.value) == "No documents to create dataset from"


@pytest.mark.parametrize(
    "as_list, clear_metadata", [(False, False), (False, True), (True, False), (True, True)]
)
def test_concatenate_datasets(
    maybe_iterable_dataset, dataset_with_converter_functions, as_list, clear_metadata
):
    # Tests four different cases of concatenation of list/dict of Datasets/IterableDatasets
    if as_list:
        # Test concatenation of list of datasets
        concatenated_dataset = concatenate_datasets(
            [
                maybe_iterable_dataset["train"],
                maybe_iterable_dataset["validation"],
                maybe_iterable_dataset["test"],
            ],
            clear_metadata=clear_metadata,
        )
    else:
        # Test concatenation of dictionary of datasets
        concatenated_dataset = concatenate_datasets(
            maybe_iterable_dataset, clear_metadata=clear_metadata
        )

    # Check correct output type
    if isinstance(maybe_iterable_dataset["train"], IterableDataset):
        # if input is IterableDataset, output should be IterableDataset
        assert isinstance(concatenated_dataset, IterableDataset)
    elif isinstance(maybe_iterable_dataset["train"], Dataset):
        # if input is Dataset, output should be Dataset
        assert isinstance(concatenated_dataset, Dataset)
    else:
        raise ValueError("Unexpected input type")

    concatenated_dataset = list(concatenated_dataset)

    for doc in concatenated_dataset:
        assert isinstance(doc, TextBasedDocument)
        if not as_list:
            # If input is dictionary, check that dataset_name is added to metadata
            assert doc.metadata["dataset_name"] is not None
            assert doc.metadata["dataset_name"] in ["test", "train", "validation"]

    assert len(concatenated_dataset) == 12

    assert [concatenated_dataset[i].id for i in [0, 8, 10]] == [
        "train_doc1",
        "val_doc1",
        "test_doc1",
    ]
    assert [doc.id for doc in concatenated_dataset[7:11]] == [
        "train_doc8",
        "val_doc1",
        "val_doc2",
        "test_doc1",
    ]


def test_concatenate_datasets_errors(dataset_with_converter_functions):
    # Test concatenation of empty datasets
    empty_dataset = list[Dataset]()
    with pytest.raises(ValueError) as excinfo:
        concatenate_datasets(empty_dataset, clear_metadata=False)
    assert str(excinfo.value) == "No datasets to concatenate"

    # Test concatenation of datasets with different document types
    dataset_with_converted_doc = dataset_with_converter_functions.to_document_type(
        TestDocumentWithLabel
    )
    with pytest.raises(ValueError) as excinfo:
        concatenate_datasets(
            [dataset_with_converter_functions, dataset_with_converted_doc], clear_metadata=False
        )
    assert str(excinfo.value) == "All datasets must have the same document type to concatenate"


def test_add_dset_name_to_document():
    # Test document having no metadata attribute
    doc = Document()
    assert not hasattr(doc, "metadata")
    with pytest.raises(ValueError) as excinfo:
        _add_dset_name_to_document(doc, "test", clear_metadata=False)
    assert (
        str(excinfo.value)
        == "Document does not have metadata attribute which required to save the dataset name: Document()"
    )

    # Test adding dataset name to document
    doc.metadata = {}
    assert hasattr(doc, "metadata")
    _add_dset_name_to_document(doc, "test_dataset_name", clear_metadata=False)
    assert doc.metadata["dataset_name"] == "test_dataset_name"

    # Test document already having dataset_name in metadata keeps the old name
    _add_dset_name_to_document(doc, "test", clear_metadata=False)
    assert doc.metadata["dataset_name"] == "test_dataset_name"
