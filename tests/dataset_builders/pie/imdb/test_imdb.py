import datasets
import pytest
from pie_core import Document

from dataset_builders.pie.imdb.imdb import Imdb
from pie_datasets import Dataset, IterableDataset
from tests.dataset_builders.common import PIE_BASE_PATH

datasets.disable_caching()

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
    return datasets.load_dataset(str(HF_DATASET_PATH), split=split)


@pytest.mark.slow
def test_hf_dataset(hf_dataset, split):
    assert hf_dataset is not None
    assert len(hf_dataset) == SPLIT_SIZES[split]


@pytest.fixture(scope="module")
def hf_dataset_fast() -> datasets.IterableDataset:
    return datasets.load_dataset(str(HF_DATASET_PATH), split=SPLIT, streaming=True).take(
        STREAM_SIZE
    )


def test_hf_dataset_fast(hf_dataset_fast):
    assert hf_dataset_fast is not None


@pytest.fixture(scope="module")
def hf_example_fast(hf_dataset_fast):
    return list(hf_dataset_fast)[0]


def test_hf_example_fast(hf_example_fast):
    assert hf_example_fast is not None
    assert hf_example_fast["text"].startswith(
        "I rented I AM CURIOUS-YELLOW from my video store because of all the controversy "
        "that surrounded it when it was first released in 1967."
    )
    assert len(hf_example_fast["text"]) == 1640
    assert hf_example_fast["label"] == 0


@pytest.fixture(scope="module")
def builder() -> BUILDER_CLASS:
    return BUILDER_CLASS()


@pytest.fixture(scope="module")
def generated_document_fast(hf_example_fast, hf_dataset_fast, builder) -> DOCUMENT_TYPE:
    generate_document_kwargs = builder._generate_document_kwargs(hf_dataset_fast) or {}
    return builder._generate_document(hf_example_fast, **generate_document_kwargs)


def test_generated_document_fast(generated_document_fast):
    assert isinstance(generated_document_fast, DOCUMENT_TYPE)
    assert generated_document_fast.text.startswith(
        "I rented I AM CURIOUS-YELLOW from my video store because of all the "
        "controversy that surrounded it when it was first released in 1967."
    )
    assert len(generated_document_fast.text) == 1640
    assert generated_document_fast.label[0].label == "neg"


def test_example_to_document_and_back_fast(
    hf_example_fast, generated_document_fast, hf_dataset_fast, builder
):
    generate_example_kwargs = builder._generate_example_kwargs(hf_dataset_fast) or {}
    hf_example_back = BUILDER_CLASS()._generate_example(
        document=generated_document_fast, **generate_example_kwargs
    )
    assert hf_example_back == hf_example_fast


@pytest.mark.slow
def test_example_to_document_and_back_all(hf_dataset, builder):
    generate_document_kwargs = builder._generate_document_kwargs(hf_dataset) or {}
    generate_example_kwargs = builder._generate_example_kwargs(hf_dataset) or {}
    for hf_ex in hf_dataset:
        doc = builder._generate_document(example=hf_ex, **generate_document_kwargs)
        hf_ex_back = builder._generate_example(document=doc, **generate_example_kwargs)
        assert hf_ex_back == hf_ex


@pytest.fixture(scope="module")
def pie_dataset(split) -> Dataset:
    return datasets.load_dataset(str(PIE_DATASET_PATH), trust_remote_code=True, split=split)


@pytest.mark.slow
def test_pie_dataset(pie_dataset, split):
    assert pie_dataset is not None
    assert len(pie_dataset) == SPLIT_SIZES[split]


@pytest.fixture(scope="module")
def pie_dataset_fast() -> IterableDataset:
    return datasets.load_dataset(
        str(PIE_DATASET_PATH), trust_remote_code=True, split=SPLIT, streaming=True
    ).take(STREAM_SIZE)


def test_pie_dataset_fast(pie_dataset_fast):
    assert pie_dataset_fast is not None


@pytest.fixture(scope="module")
def document_fast(pie_dataset_fast) -> DOCUMENT_TYPE:
    doc = list(pie_dataset_fast)[0]
    # we can not assert the real document type because it may come from a dataset loading script
    # downloaded to a temporary directory and thus have a different type object, although it is
    # semantically the same
    assert isinstance(doc, Document)
    casted = doc.as_type(DOCUMENT_TYPE)
    return casted


def test_compare_document_and_generated_document_fast(document_fast, generated_document_fast):
    assert document_fast == generated_document_fast


def test_dataset_with_converted_documents_fast(
    pie_dataset_fast, document_fast, converted_document_type
):
    dataset_with_converted_documents = pie_dataset_fast.to_document_type(converted_document_type)
    assert dataset_with_converted_documents is not None
    # check first document
    doc_converted = list(dataset_with_converted_documents)[0]
    assert isinstance(doc_converted, converted_document_type)
    doc_casted = document_fast.as_type(converted_document_type)
    assert doc_converted == doc_casted


@pytest.mark.slow
def test_dataset_with_converted_documents_all(pie_dataset, converted_document_type):
    dataset_with_converted_documents = pie_dataset.to_document_type(converted_document_type)
    assert dataset_with_converted_documents is not None
    # check type of all documents
    for converted_doc in dataset_with_converted_documents:
        assert isinstance(converted_doc, converted_document_type)
