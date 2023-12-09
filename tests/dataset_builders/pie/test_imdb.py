import pytest
from datasets import disable_caching, load_dataset
from pytorch_ie.core import Document

from dataset_builders.pie.imdb.imdb import Imdb, document_to_example
from pie_datasets import DatasetDict
from tests import FIXTURES_ROOT
from tests.dataset_builders.common import PIE_BASE_PATH

disable_caching()

DATASET_NAME = "imdb"
BUILDER_CLASS = Imdb
DOCUMENT_TYPE = BUILDER_CLASS.DOCUMENT_TYPE
SPLIT_SIZES = {"test": 25000, "train": 25000, "unsupervised": 50000}
HF_DATASET_PATH = BUILDER_CLASS.BASE_DATASET_PATH
PIE_DATASET_PATH = PIE_BASE_PATH / DATASET_NAME
DATA_PATH = FIXTURES_ROOT / "dataset_builders" / "squad_v2.zip"


@pytest.fixture(scope="module", params=list(BUILDER_CLASS.DOCUMENT_CONVERTERS))
def converted_document_type(request):
    return request.param


@pytest.fixture(scope="module", params=list(SPLIT_SIZES))
def split(request):
    return request.param


@pytest.fixture(scope="module")
def hf_dataset():
    return load_dataset(str(HF_DATASET_PATH), data_dir=DATA_PATH)


def test_hf_dataset(hf_dataset):
    assert hf_dataset is not None
    assert {name: len(ds) for name, ds in hf_dataset.items()} == SPLIT_SIZES


@pytest.fixture(scope="module")
def hf_example(hf_dataset, split):
    return hf_dataset[split][0]


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
def generate_document_kwargs(hf_dataset, split) -> dict:
    return BUILDER_CLASS()._generate_document_kwargs(hf_dataset[split]) or {}


@pytest.fixture(scope="module")
def generated_document(hf_example, generate_document_kwargs) -> DOCUMENT_TYPE:
    return BUILDER_CLASS()._generate_document(hf_example, **generate_document_kwargs)


def test_generated_document(generated_document, split):
    assert isinstance(generated_document, DOCUMENT_TYPE)
    if split == "train":
        assert generated_document.text.startswith(
            "I rented I AM CURIOUS-YELLOW from my video store because of all the controversy that surrounded it when it was first released in 1967."
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


@pytest.fixture(scope="module")
def hf_example_back(generated_document, generate_document_kwargs):
    return document_to_example(document=generated_document, **generate_document_kwargs)


def test_example_to_document_and_back(hf_example, hf_example_back):
    assert hf_example_back == hf_example


@pytest.mark.slow
def test_example_to_document_and_back_all(hf_dataset, generate_document_kwargs, split):
    for hf_ex in hf_dataset[split]:
        doc = BUILDER_CLASS()._generate_document(example=hf_ex, **generate_document_kwargs)
        hf_ex_back = document_to_example(document=doc, **generate_document_kwargs)
        assert hf_ex_back == hf_ex


@pytest.fixture(scope="module")
def dataset() -> DatasetDict:
    return DatasetDict.load_dataset(str(PIE_DATASET_PATH))


def test_pie_dataset(dataset):
    assert dataset is not None
    assert {name: len(ds) for name, ds in dataset.items()} == SPLIT_SIZES


@pytest.fixture(scope="module")
def document(dataset, split) -> DOCUMENT_TYPE:
    doc = dataset[split][0]
    # we can not assert the real document type because it may come from a dataset loading script
    # downloaded to a temporary directory and thus have a different type object, although it is
    # semantically the same
    assert isinstance(doc, Document)
    casted = doc.as_type(DOCUMENT_TYPE)
    return casted


def test_compare_document_and_generated_document(document, generated_document):
    assert document == generated_document


@pytest.fixture(scope="module")
def dataset_with_extractive_qa_documents(dataset, converted_document_type) -> DatasetDict:
    return dataset.to_document_type(converted_document_type)


def test_dataset_with_extractive_qa_documents(
    dataset_with_extractive_qa_documents, document, split, converted_document_type
):
    assert dataset_with_extractive_qa_documents is not None
    doc = dataset_with_extractive_qa_documents[split][0]
    assert isinstance(doc, converted_document_type)
    doc_casted = document.as_type(converted_document_type)
    assert doc == doc_casted
