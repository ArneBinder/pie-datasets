import pytest
from datasets import disable_caching, load_dataset

from dataset_builders.pie.scientific_papers.scientific_papers import ScientificPapers
from pie_datasets import DatasetDict
from tests.dataset_builders.common import PIE_BASE_PATH

disable_caching()

DATASET_NAME = "scientific_papers"
BUILDER_CLASS = ScientificPapers
SPILT_SIZES = {
    "arxiv": {"train": 203037, "validation": 6436, "test": 6440},
    "pubmed": {"train": 119924, "validation": 6633, "test": 6658},
}
DOCUMENT_TYPE = BUILDER_CLASS.DOCUMENT_TYPE
HF_DATASET_PATH = BUILDER_CLASS.BASE_DATASET_PATH
PIE_DATASET_PATH = PIE_BASE_PATH / DATASET_NAME


@pytest.fixture(scope="module", params=["arxiv", "pubmed"])
def dataset_variant(request):
    return request.param


@pytest.fixture(scope="module")
def hf_dataset(dataset_variant):
    return load_dataset(DATASET_NAME, dataset_variant)


@pytest.fixture(scope="module")
def hf_example(dataset_variant):
    return load_dataset(DATASET_NAME, dataset_variant)["train"][0]


@pytest.fixture(scope="module")
def generate_document_kwargs(hf_dataset, dataset_variant):
    return (
        BUILDER_CLASS(config_name=dataset_variant)._generate_document_kwargs(hf_dataset["train"])
        or {}
    )


def test_hf_dataset(hf_dataset, dataset_variant):
    assert hf_dataset is not None

    splits = {name: len(ds) for name, ds in hf_dataset.items()}
    assert splits == SPILT_SIZES[dataset_variant]


def test_hf_example(hf_example):
    assert hf_example is not None
    assert isinstance(hf_example, dict)
    assert "article" in hf_example
    assert "abstract" in hf_example
    assert "section_names" in hf_example


def test_generate_document_kwargs(hf_dataset, generate_document_kwargs):
    assert generate_document_kwargs is not None
    assert isinstance(generate_document_kwargs, dict)


def test_generate_document(hf_example, generate_document_kwargs, dataset_variant):
    doc = BUILDER_CLASS(config_name=dataset_variant)._generate_document(
        hf_example, **generate_document_kwargs
    )
    assert doc is not None
    assert isinstance(doc, DOCUMENT_TYPE)
    assert doc.text is not None
    assert doc.abstract is not None
    assert doc.section_names is not None


def test_generate_example(hf_example, generate_document_kwargs, dataset_variant):
    doc = BUILDER_CLASS(config_name=dataset_variant)._generate_document(
        hf_example, **generate_document_kwargs
    )
    example = BUILDER_CLASS(config_name=dataset_variant)._generate_example(doc)
    assert example is not None
    assert isinstance(example, dict)
