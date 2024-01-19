import pytest
from datasets import disable_caching, load_dataset

from dataset_builders.pie.conll2012_ontonotesv5.conll2012_ontonotesv5 import (
    Conll2012Ontonotesv5,
)
from pie_datasets import load_dataset as load_pie_dataset
from tests.dataset_builders.common import PIE_BASE_PATH

disable_caching()

DATASET_NAME = "conll2012_ontonotesv5"
BUILDER_CLASS = Conll2012Ontonotesv5
DOCUMENT_TYPE = BUILDER_CLASS.DOCUMENT_TYPE
HF_DATASET_PATH = BUILDER_CLASS.BASE_DATASET_PATH
PIE_DATASET_PATH = PIE_BASE_PATH / DATASET_NAME
STREAM_SIZE = 5
SPLIT_NAMES = {"train", "validation", "test"}


@pytest.fixture(scope="module", params=[config.name for config in BUILDER_CLASS.BUILDER_CONFIGS])
def dataset_variant(request):
    return request.param


@pytest.fixture(params=SPLIT_NAMES, scope="module")
def split_name(request):
    return request.param


@pytest.fixture(scope="module")
def hf_dataset(dataset_variant, split_name):
    dataset = load_dataset(
        BUILDER_CLASS.BASE_DATASET_PATH, name=dataset_variant, split=split_name, streaming=True
    )
    dataset_head = dataset.take(STREAM_SIZE)
    return list(dataset_head)


def test_hf_dataset(hf_dataset, dataset_variant, split_name):
    assert hf_dataset is not None


@pytest.fixture(scope="module")
def pie_dataset(dataset_variant, split_name):
    dataset = load_pie_dataset(
        str(PIE_DATASET_PATH), name=dataset_variant, split=split_name, streaming=True
    )
    dataset_head = dataset.take(STREAM_SIZE)
    return list(dataset_head)


def test_pie_dataset(pie_dataset, dataset_variant, split_name):
    assert pie_dataset is not None


@pytest.fixture(scope="module")
def hf_example(hf_dataset):
    return hf_dataset[0]


@pytest.fixture(scope="module")
def pie_example(pie_dataset):
    return pie_dataset[0]
