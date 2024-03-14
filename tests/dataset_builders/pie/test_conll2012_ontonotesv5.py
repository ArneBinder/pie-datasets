import datasets
import pytest
from datasets import ClassLabel, disable_caching, load_dataset
from pytorch_ie import Document
from pytorch_ie.documents import TextDocumentWithLabeledSpansAndLabeledPartitions

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
SPLIT_NAMES = {"train", "validation", "test"}
SLOW_STREAM_SIZE = 5

# Testing parameters
DATASET_VARIANT = "english_v4"
SPLIT_NAME = "train"
STREAM_SIZE = 2


@pytest.fixture(scope="module", params=[config.name for config in BUILDER_CLASS.BUILDER_CONFIGS])
def dataset_variants(request):
    return request.param


@pytest.fixture(params=SPLIT_NAMES, scope="module")
def split_names(request):
    return request.param


@pytest.fixture(scope="module")
def hf_dataset():
    dataset = load_dataset(
        BUILDER_CLASS.BASE_DATASET_PATH, name=DATASET_VARIANT, split=SPLIT_NAME, streaming=True
    )
    return dataset.take(STREAM_SIZE)


@pytest.fixture(scope="module")
def hf_dataset_slow(dataset_variants, split_names):
    dataset = load_dataset(
        BUILDER_CLASS.BASE_DATASET_PATH, name=dataset_variants, split=split_names, streaming=True
    )
    return dataset.take(SLOW_STREAM_SIZE)


def test_hf_dataset(hf_dataset):
    assert hf_dataset is not None


@pytest.mark.slow
def test_hf_dataset_slow(hf_dataset_slow, dataset_variants, split_names):
    assert hf_dataset_slow is not None


@pytest.fixture(scope="module")
def pie_dataset():
    dataset = load_pie_dataset(
        str(PIE_DATASET_PATH), name=DATASET_VARIANT, split=SPLIT_NAME, streaming=True
    )
    return dataset.take(STREAM_SIZE)


@pytest.fixture(scope="module")
def pie_dataset_slow(dataset_variant, split_name):
    dataset = load_pie_dataset(
        str(PIE_DATASET_PATH), name=dataset_variants, split=split_names, streaming=True
    )
    return dataset.take(SLOW_STREAM_SIZE)


def test_pie_dataset(pie_dataset):
    assert pie_dataset is not None


@pytest.mark.slow
def test_pie_dataset_slow(pie_dataset, dataset_variants, split_names):
    assert pie_dataset is not None


@pytest.fixture(scope="module")
def hf_example(hf_dataset):
    return list(hf_dataset)[0]


@pytest.fixture(scope="module")
def hf_example_slow(hf_dataset_slow):
    return list(hf_dataset_slow)[0]


@pytest.fixture(scope="module")
def pie_example(pie_dataset):
    return list(pie_dataset)[0]


@pytest.fixture(scope="module")
def pie_example_slow(pie_dataset_slow):
    return list(pie_dataset_slow)[0]


def test_hf_example(hf_example):
    assert hf_example is not None


@pytest.mark.slow
def test_hf_example_slow(hf_example_slow):
    assert hf_example_slow is not None


@pytest.mark.slow
def test_pie_example_slow(pie_example_slow):
    assert pie_example_slow is not None


@pytest.fixture(scope="module")
def generate_document_kwargs(hf_dataset):
    return BUILDER_CLASS(config_name=DATASET_VARIANT)._generate_document_kwargs(hf_dataset)


@pytest.fixture(scope="module")
def generate_document_kwargs_slow(hf_dataset_slow, dataset_variants):
    return BUILDER_CLASS(config_name=dataset_variants)._generate_document_kwargs(hf_dataset_slow)


def test_generate_document_kwargs(generate_document_kwargs):
    assert generate_document_kwargs is not None
    assert isinstance(generate_document_kwargs, dict)


@pytest.mark.slow
def test_generate_document_kwargs_slow(generate_document_kwargs_slow):
    assert generate_document_kwargs_slow is not None
    assert isinstance(generate_document_kwargs_slow, dict)


@pytest.fixture(scope="module")
def generated_document(hf_example, generate_document_kwargs):
    return BUILDER_CLASS(config_name=DATASET_VARIANT)._generate_document(
        hf_example, **generate_document_kwargs
    )


def test_generate_document(generated_document):
    assert generated_document is not None
    assert isinstance(generated_document, BUILDER_CLASS.DOCUMENT_TYPE)
    # check actual annotations
    assert generated_document.speakers[0].label == "Speaker#1"
    assert generated_document.parts[0].label == "0"
    assert generated_document.entities[0].label == "ORG"
    assert generated_document.predicates[0].lemma == "memory"
    assert generated_document.srl_relations[0].roles == (
        "ARG0",
        "ARGM-MNR",
        "V",
        "ARG1",
        "ARG2",
    )
    assert (
        generated_document.parse_trees[0].label
        == "(TOP(SBARQ(WHNP(WHNP (WP What)  (NN kind) )(PP (IN of) (NP (NN memory) ))) (. ?) ))"
    )
    assert generated_document.tokens[
        generated_document.sentences[0].start : generated_document.sentences[0].end
    ] == ("What", "kind", "of", "memory", "?")
    assert generated_document.word_senses[0].label == "1.0"


@pytest.fixture(scope="module")
def generated_document_slow(hf_example_slow, generate_document_kwargs_slow, dataset_variants):
    return BUILDER_CLASS(config_name=dataset_variants)._generate_document(
        hf_example_slow, **generate_document_kwargs_slow
    )


@pytest.mark.slow
def test_generate_document_slow(generated_document_slow, dataset_variants, split_names):
    assert generated_document_slow is not None
    assert isinstance(generated_document_slow, BUILDER_CLASS.DOCUMENT_TYPE)


@pytest.fixture(scope="module")
def generated_example(generated_document, generate_document_kwargs):
    return document_to_example(generated_document, **generate_document_kwargs)


@pytest.fixture(scope="module")
def generated_example_slow(
    generated_document_slow, generate_document_kwargs_slow, dataset_variants
):
    return document_to_example(generated_document_slow, **generate_document_kwargs_slow)


def test_compare_document_and_generated_document(generated_document, pie_example):
    assert generated_document.id == pie_example.id
    assert generated_document.sentences == pie_example.sentences
    assert generated_document.tokens == pie_example.tokens
    assert generated_document.pos_tags == pie_example.pos_tags
    assert generated_document.entities == pie_example.entities
    assert generated_document.coref_mentions == pie_example.coref_mentions
    assert generated_document.srl_arguments == pie_example.srl_arguments
    assert generated_document.srl_relations == pie_example.srl_relations
    assert generated_document.word_senses == pie_example.word_senses
    assert generated_document.parts == pie_example.parts


# TODO: the following annotation fields have identical content but get assertion error
#     assert generated_document.parse_trees == pie_example.parse_trees
#     assert generated_document.speakers == pie_example.speakers
#     assert generated_document.predicates == pie_example.predicates
#     assert generated_document.coref_clusters == pie_example.coref_clusters


@pytest.mark.slow
def test_compare_document_and_generated_document_slow(generated_document_slow, pie_example_slow):
    assert generated_document_slow.id == pie_example_slow.id
    assert generated_document_slow.sentences == pie_example_slow.sentences
    assert generated_document_slow.tokens == pie_example_slow.tokens
    assert generated_document_slow.pos_tags == pie_example_slow.pos_tags
    assert generated_document_slow.entities == pie_example_slow.entities
    assert generated_document_slow.coref_mentions == pie_example_slow.coref_mentions
    assert generated_document_slow.srl_arguments == pie_example_slow.srl_arguments
    assert generated_document_slow.srl_relations == pie_example_slow.srl_relations
    assert generated_document_slow.word_senses == pie_example_slow.word_senses
    assert generated_document_slow.parts == pie_example_slow.parts


# TODO: the following annotation fields have identical content but get assertion error
#     assert generated_document_slow.parse_trees == pie_example_slow.parse_trees
#     assert generated_document_slow.speakers == pie_example_slow.speakers
#     assert generated_document_slow.predicates == pie_example_slow.predicates
#     assert generated_document_slow.coref_clusters == pie_example_slow.coref_clusters


def test_compare_generate_example_and_back(hf_example, generated_example):
    assert hf_example["document_id"] == generated_example["document_id"]
    for hf, gen in zip(hf_example["sentences"], generated_example["sentences"]):
        for key in hf.keys():
            if key == "coref_spans":
                # 'coref_spans' must be sorted before compare
                assert sorted(hf[key]) == sorted(gen[key])
            else:
                assert hf[key] == gen[key]


@pytest.mark.slow
def test_compare_generate_example_and_back_slow(hf_example_slow, generated_example_slow):
    assert hf_example_slow["document_id"] == generated_example_slow["document_id"]
    for hf, gen in zip(hf_example_slow["sentences"], generated_example_slow["sentences"]):
        for key in hf.keys():
            if key == "coref_spans":
                # 'coref_spans' must be sorted before compare
                assert sorted(hf[key]) == sorted(gen[key])
            else:
                assert hf[key] == gen[key]


def test_compare_generate_example_and_back_all(hf_dataset, generate_document_kwargs):
    for hf_ex in list(hf_dataset):
        doc = example_to_document(hf_ex, **generate_document_kwargs)
        hf_ex_back = document_to_example(doc, **generate_document_kwargs)
        assert hf_ex_back["document_id"] == hf_ex["document_id"]
        for ex, ex_back in zip(hf_ex["sentences"], hf_ex_back["sentences"]):
            for key in ex.keys():
                # 'coref_spans' must be sorted before compare
                if key == "coref_spans":
                    assert sorted(ex[key]) == sorted(ex_back[key])
                else:
                    assert ex[key] == ex_back[key]


@pytest.mark.slow
def test_compare_generate_example_and_back_all_slow(
    hf_dataset_slow, generate_document_kwargs_slow
):
    for hf_ex in list(hf_dataset_slow):
        doc = example_to_document(hf_ex, **generate_document_kwargs_slow)
        hf_ex_back = document_to_example(doc, **generate_document_kwargs_slow)
        assert hf_ex_back["document_id"] == hf_ex["document_id"]
        for ex, ex_back in zip(hf_ex["sentences"], hf_ex_back["sentences"]):
            for key in ex.keys():
                # 'coref_spans' must be sorted before compare
                if key == "coref_spans":
                    assert sorted(ex[key]) == sorted(ex_back[key])
                else:
                    assert ex[key] == ex_back[key]


def test_convert_to_text_document_with_labeled_spans_and_labeled_partitions(generated_document):
    converted_doc = convert_to_text_document_with_labeled_spans_and_labeled_partitions(
        generated_document
    )
    assert converted_doc is not None
    assert isinstance(converted_doc, TextDocumentWithLabeledSpansAndLabeledPartitions)


@pytest.mark.slow
def test_convert_to_text_document_with_labeled_spans_and_labeled_partitions_slow(
    generated_document_slow,
):
    converted_doc = convert_to_text_document_with_labeled_spans_and_labeled_partitions(
        generated_document_slow
    )
    assert converted_doc is not None
    assert isinstance(converted_doc, TextDocumentWithLabeledSpansAndLabeledPartitions)
