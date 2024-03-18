import datasets
import pytest
from datasets import disable_caching, load_dataset
from pytorch_ie.core import Document

from dataset_builders.pie.imdb.imdb import Imdb
from pie_datasets import Dataset
from tests.dataset_builders.common import PIE_BASE_PATH

disable_caching()

DATASET_NAME = "imdb"
BUILDER_CLASS = Imdb
DOCUMENT_TYPE = BUILDER_CLASS.DOCUMENT_TYPE
SPLIT_SIZES = {"test": 25000, "train": 25000, "unsupervised": 50000}
HF_DATASET_PATH = BUILDER_CLASS.BASE_DATASET_PATH
PIE_DATASET_PATH = PIE_BASE_PATH / DATASET_NAME

# fast testing parameters
SPLIT = "train"
STREAM_SIZE = 3


@pytest.fixture(scope="module", params=list(BUILDER_CLASS.DOCUMENT_CONVERTERS))
def converted_document_type(request):
    return request.param


@pytest.fixture(scope="module", params=list(SPLIT_SIZES))
def split(request):
    return request.param


@pytest.fixture(scope="module")
def hf_dataset(split) -> datasets.Dataset:
    return load_dataset(str(HF_DATASET_PATH), split=split)


def test_hf_dataset(hf_dataset, split):
    assert hf_dataset is not None
    assert len(hf_dataset) == SPLIT_SIZES[split]


@pytest.fixture(scope="module")
def hf_example(hf_dataset):
    return hf_dataset[0]


def test_hf_example(hf_example, split):
    assert hf_example is not None
    if split == "train":
        assert hf_example["text"].startswith(
            "I rented I AM CURIOUS-YELLOW from my video store because of all the controversy "
            "that surrounded it when it was first released in 1967."
        )
        assert len(hf_example["text"]) == 1640
        assert hf_example["label"] == 0
    elif split == "test":
        assert hf_example["text"].startswith("I love sci-fi and am willing to put up with a lot.")
        assert len(hf_example["text"]) == 1386
        assert hf_example["label"] == 0
    elif split == "unsupervised":
        assert hf_example["text"].startswith("This is just a precious little diamond.")
        assert len(hf_example["text"]) == 892
        assert hf_example["label"] == -1
    else:
        raise ValueError(f"Unknown split: {split}")


@pytest.fixture(scope="module")
def builder() -> BUILDER_CLASS:
    return BUILDER_CLASS()


@pytest.fixture(scope="module")
def generate_document_kwargs(hf_dataset, builder) -> dict:
    return builder._generate_document_kwargs(hf_dataset) or {}


@pytest.fixture(scope="module")
def generate_example_kwargs(hf_dataset, builder) -> dict:
    return builder._generate_example_kwargs(hf_dataset) or {}


@pytest.fixture(scope="module")
def generated_document(hf_example, generate_document_kwargs, builder) -> DOCUMENT_TYPE:
    return builder._generate_document(hf_example, **generate_document_kwargs)


def test_generated_document(generated_document, split):
    assert isinstance(generated_document, DOCUMENT_TYPE)
    if split == "train":
        assert generated_document.text.startswith(
            "I rented I AM CURIOUS-YELLOW from my video store because of all the "
            "controversy that surrounded it when it was first released in 1967."
        )
        assert len(generated_document.text) == 1640
        assert generated_document.label[0].label == "neg"
    elif split == "test":
        assert generated_document.text.startswith(
            "I love sci-fi and am willing to put up with a lot."
        )
        assert len(generated_document.text) == 1386
        assert generated_document.label[0].label == "neg"
    elif split == "unsupervised":
        assert generated_document.text.startswith("This is just a precious little diamond.")
        assert len(generated_document.text) == 892
        assert len(generated_document.label) == 0
    else:
        raise ValueError(f"Unknown split: {split}")


def test_example_to_document_and_back(hf_example, generated_document, generate_example_kwargs):
    hf_example_back = BUILDER_CLASS()._generate_example(
        document=generated_document, **generate_example_kwargs
    )
    assert hf_example_back == hf_example


@pytest.mark.slow
def test_example_to_document_and_back_all(
    hf_dataset, generate_document_kwargs, generate_example_kwargs, builder
):
    for hf_ex in hf_dataset:
        doc = builder._generate_document(example=hf_ex, **generate_document_kwargs)
        hf_ex_back = builder._generate_example(document=doc, **generate_example_kwargs)
        assert hf_ex_back == hf_ex


@pytest.fixture(scope="module")
def pie_dataset(split) -> Dataset:
    return load_dataset(str(PIE_DATASET_PATH), split=split)


def test_pie_dataset(pie_dataset, split):
    assert pie_dataset is not None
    assert len(pie_dataset) == SPLIT_SIZES[split]


@pytest.fixture(scope="module")
def document(pie_dataset) -> DOCUMENT_TYPE:
    doc = pie_dataset[0]
    # we can not assert the real document type because it may come from a dataset loading script
    # downloaded to a temporary directory and thus have a different type object, although it is
    # semantically the same
    assert isinstance(doc, Document)
    casted = doc.as_type(DOCUMENT_TYPE)
    return casted


def test_compare_document_and_generated_document(document, generated_document):
    assert document == generated_document


def test_dataset_with_converted_documents(pie_dataset, document, converted_document_type):
    dataset_with_converted_documents = pie_dataset.to_document_type(converted_document_type)
    assert dataset_with_converted_documents is not None
    doc_converted = dataset_with_converted_documents[0]
    assert isinstance(doc_converted, converted_document_type)
    doc_casted = document.as_type(converted_document_type)
    assert doc_converted == doc_casted
