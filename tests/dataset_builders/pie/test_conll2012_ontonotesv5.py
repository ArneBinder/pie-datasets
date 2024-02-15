import datasets
import pytest
from datasets import ClassLabel, disable_caching, load_dataset
from pytorch_ie import Document
from pytorch_ie.documents import (
    TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions,
)

from dataset_builders.pie.conll2012_ontonotesv5.conll2012_ontonotesv5 import (
    Conll2012Ontonotesv5,
    convert_to_text_document_with_labeled_spans_and_labeled_partitions,
    document_to_example,
    example_to_document,
)
from pie_datasets import DatasetDict
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
    return dataset.take(STREAM_SIZE)
    # dataset_head = dataset.take(STREAM_SIZE)
    # return list(dataset_head)


def test_hf_dataset(hf_dataset, dataset_variant, split_name):
    assert hf_dataset is not None


@pytest.fixture(scope="module")
def pie_dataset(dataset_variant, split_name):
    dataset = load_pie_dataset(
        str(PIE_DATASET_PATH), name=dataset_variant, split=split_name, streaming=True
    )
    return dataset.take(STREAM_SIZE)
    # dataset_head = dataset.take(STREAM_SIZE)
    # return list(dataset_head)


def test_pie_dataset(pie_dataset, dataset_variant, split_name):
    assert pie_dataset is not None


@pytest.fixture(scope="module")
def hf_example(hf_dataset):
    return list(hf_dataset)[0]


@pytest.fixture(scope="module")
def pie_example(pie_dataset):
    return list(pie_dataset)[0]


def test_hf_example(hf_example):
    assert hf_example is not None


def test_pie_example(pie_example):
    assert pie_example is not None


@pytest.fixture(scope="module")
def entity_labels(hf_dataset, dataset_variant, split_name):
    return hf_dataset.features["sentences"][0]["named_entities"].feature


@pytest.fixture(scope="module")
def pos_tag_labels(hf_dataset, dataset_variant, split_name):
    pos_tags_feature = hf_dataset.features["sentences"][0]["pos_tags"].feature
    return pos_tags_feature if isinstance(pos_tags_feature, datasets.ClassLabel) else None


def test_compare_examples(hf_example, pie_example):
    # compare annotations between HF and PIE
    assert hf_example is not None
    assert pie_example is not None
    assert hf_example["document_id"] == pie_example.id
    # pie_example.sentences
    # pie_example.tokens
    # pie_example.pos_tags
    # pie_example.parse_trees
    # pie_example.speakers
    # pie_example.entities
    # pie_example.predicates
    # pie_example.coref_mentions
    # pie_example.coref_clusters
    # pie_example.srl_arguments
    # pie_example.srl_relations
    # pie_example.word_senses
    # pie_example.parts


@pytest.fixture(scope="module")
def generate_document_kwargs(hf_dataset, entity_labels, pos_tag_labels, dataset_variant):
    # return BUILDER_CLASS(config_name=dataset_variant)._generate_document_kwargs(hf_dataset) or {}
    return dict(entity_labels=entity_labels, pos_tag_labels=pos_tag_labels)


def test_generate_document_kwargs(generate_document_kwargs):
    assert generate_document_kwargs is not None
    assert isinstance(generate_document_kwargs, dict)


@pytest.fixture(scope="module")
def generated_document(hf_example, generate_document_kwargs, dataset_variant):
    return example_to_document(hf_example, **generate_document_kwargs)


def test_generate_document(generated_document):
    assert generated_document is not None
    assert isinstance(generated_document, BUILDER_CLASS.DOCUMENT_TYPE)


def test_compare_document_and_generated_document(generated_document, pie_example):
    assert generated_document == pie_example


@pytest.fixture(scope="module")
def generated_example(generated_document, generate_document_kwargs, dataset_variant):
    return document_to_example(generated_document, **generate_document_kwargs)


def test_generate_example(generated_example):
    assert generated_example is not None
    assert isinstance(generated_example, dict)


def test_compare_generate_example_and_back(hf_example, generated_example):
    assert hf_example == generated_example


# def test_compare_generate_example_and_back_all


def test_convert_to_text_document_with_labeled_spans_and_labeled_partitions(generated_document):
    converted_doc = convert_to_text_document_with_labeled_spans_and_labeled_partitions(
        generated_document
    )
    assert converted_doc is not None
    assert isinstance(
        converted_doc, TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions
    )
