import pytest
from datasets import disable_caching, load_dataset
from pytorch_ie.documents import TextDocumentWithLabeledSpansAndLabeledPartitions

from dataset_builders.pie.conll2012_ontonotesv5.conll2012_ontonotesv5 import (
    Conll2012Ontonotesv5,
    convert_to_text_document_with_labeled_spans_and_labeled_partitions,
    document_to_example,
    example_to_document,
)
from pie_datasets import load_dataset as load_pie_dataset
from tests.dataset_builders.common import PIE_BASE_PATH

disable_caching()

DATASET_NAME = "conll2012_ontonotesv5"
BUILDER_CLASS = Conll2012Ontonotesv5
DOCUMENT_TYPE = BUILDER_CLASS.DOCUMENT_TYPE
HF_DATASET_PATH = BUILDER_CLASS.BASE_DATASET_PATH
PIE_DATASET_PATH = PIE_BASE_PATH / DATASET_NAME
SPLIT_NAMES = {"train", "validation", "test"}
SLOW_STREAM_SIZE = None

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
def hf_dataset_extract():
    dataset = load_dataset(
        BUILDER_CLASS.BASE_DATASET_PATH, name=DATASET_VARIANT, split=SPLIT_NAME, streaming=True
    )
    return dataset.take(STREAM_SIZE)


@pytest.fixture(scope="module")
def hf_dataset(dataset_variants, split_names):
    dataset = load_dataset(
        BUILDER_CLASS.BASE_DATASET_PATH,
        name=dataset_variants,
        split=split_names,
        streaming=SLOW_STREAM_SIZE is not None,
    )
    if SLOW_STREAM_SIZE is not None:
        dataset = dataset.take(SLOW_STREAM_SIZE)

    return dataset


def test_hf_dataset_extract(hf_dataset_extract):
    assert hf_dataset_extract is not None


@pytest.mark.slow
def test_hf_dataset(hf_dataset, dataset_variants, split_names):
    assert hf_dataset is not None


@pytest.fixture(scope="module")
def pie_dataset_extract():
    dataset = load_pie_dataset(
        str(PIE_DATASET_PATH), name=DATASET_VARIANT, split=SPLIT_NAME, streaming=True
    )
    return dataset.take(STREAM_SIZE)


@pytest.fixture(scope="module")
def pie_dataset(dataset_variants, split_names):
    dataset = load_pie_dataset(
        str(PIE_DATASET_PATH),
        name=dataset_variants,
        split=split_names,
        streaming=SLOW_STREAM_SIZE is not None,
    )
    if SLOW_STREAM_SIZE is not None:
        dataset = dataset.take(SLOW_STREAM_SIZE)

    return dataset


def test_pie_dataset_extract(pie_dataset_extract):
    assert pie_dataset_extract is not None


@pytest.mark.slow
def test_pie_dataset(pie_dataset, dataset_variants, split_names):
    assert pie_dataset is not None


@pytest.fixture(scope="module")
def hf_example(hf_dataset_extract):
    return list(hf_dataset_extract)[0]


@pytest.fixture(scope="module")
def pie_example(pie_dataset_extract):
    return list(pie_dataset_extract)[0]


def test_hf_example(hf_example):
    assert hf_example is not None


@pytest.mark.slow
def test_pie_example(pie_example):
    assert pie_example is not None


@pytest.fixture(scope="module")
def generate_document_kwargs(hf_dataset_extract):
    return BUILDER_CLASS(config_name=DATASET_VARIANT)._generate_document_kwargs(hf_dataset_extract)


def test_generate_document_kwargs(generate_document_kwargs):
    assert generate_document_kwargs is not None
    assert isinstance(generate_document_kwargs, dict)


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
def generated_example(generated_document, generate_document_kwargs):
    return document_to_example(generated_document, **generate_document_kwargs)


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
def test_compare_generate_example_and_back_all_slow(hf_dataset, generate_document_kwargs):
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


def test_convert_to_text_document_with_labeled_spans_and_labeled_partitions(generated_document):
    converted_doc = convert_to_text_document_with_labeled_spans_and_labeled_partitions(
        generated_document
    )
    assert converted_doc is not None
    assert isinstance(converted_doc, TextDocumentWithLabeledSpansAndLabeledPartitions)
