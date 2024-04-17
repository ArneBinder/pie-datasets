import json
from typing import Any, List, Sequence

import pytest
from datasets import disable_caching, load_dataset
from pytorch_ie import Annotation
from pytorch_ie.annotations import BinaryRelation, LabeledMultiSpan, LabeledSpan, Span
from pytorch_ie.documents import TextDocumentWithLabeledSpansAndLabeledPartitions

from dataset_builders.pie.conll2012_ontonotesv5.conll2012_ontonotesv5 import (
    Conll2012Ontonotesv5,
    convert_to_text_document_with_labeled_spans_and_labeled_partitions,
    document_to_example,
    example_to_document,
)
from pie_datasets import load_dataset as load_pie_dataset
from pie_datasets.builders.brat import BratAttribute
from tests.dataset_builders.common import HF_DS_FIXTURE_DATA_PATH, PIE_BASE_PATH

disable_caching()

DATASET_NAME = "conll2012_ontonotesv5"
BUILDER_CLASS = Conll2012Ontonotesv5
DOCUMENT_TYPE = BUILDER_CLASS.DOCUMENT_TYPE
HF_DATASET_PATH = BUILDER_CLASS.BASE_DATASET_PATH
PIE_DATASET_PATH = PIE_BASE_PATH / DATASET_NAME
SPLIT_NAMES = {"train", "validation", "test"}
SPLIT_SIZES = {
    "english_v4": {"train": 1940, "validation": 222, "test": 222},
    "chinese_v4": {"train": 1391, "validation": 172, "test": 166},
    "arabic_v4": {"train": 359, "validation": 44, "test": 44},
    "english_v12": {"train": 10539, "validation": 1370, "test": 1200},
}

# Testing parameters
DATASET_VARIANT = "english_v4"
SPLIT_NAME = "train"
HF_EXAMPLE_FIXTURE_PATH = HF_DS_FIXTURE_DATA_PATH / DATASET_NAME / "example.json"
STREAM_SIZE = 2


@pytest.fixture(scope="module", params=[config.name for config in BUILDER_CLASS.BUILDER_CONFIGS])
def dataset_variant(request):
    return request.param


@pytest.fixture(params=SPLIT_NAMES, scope="module")
def split_name(request):
    return request.param


@pytest.fixture(scope="module")
def hf_dataset_extract():
    dataset = load_dataset(
        BUILDER_CLASS.BASE_DATASET_PATH, name=DATASET_VARIANT, split=SPLIT_NAME, streaming=True
    )
    return dataset.take(STREAM_SIZE)


@pytest.fixture(scope="module")
def hf_dataset_all(dataset_variant, split_name):
    dataset = load_dataset(
        BUILDER_CLASS.BASE_DATASET_PATH,
        name=dataset_variant,
        split=split_name,
        streaming=False,
    )

    return dataset


def test_hf_dataset_extract(hf_dataset_extract):
    assert hf_dataset_extract is not None


@pytest.mark.slow
def test_hf_dataset_all(hf_dataset_all, dataset_variant, split_name):
    assert hf_dataset_all is not None
    assert len(hf_dataset_all) == SPLIT_SIZES[dataset_variant][split_name]


@pytest.fixture(scope="module")
def pie_dataset_extract():
    dataset = load_pie_dataset(
        str(PIE_DATASET_PATH), name=DATASET_VARIANT, split=SPLIT_NAME, streaming=True
    )
    return dataset.take(STREAM_SIZE)


@pytest.fixture(scope="module")
def pie_dataset_all(dataset_variant, split_name):
    dataset = load_pie_dataset(
        str(PIE_DATASET_PATH),
        name=dataset_variant,
        split=split_name,
        streaming=False,
    )

    return dataset


def test_pie_dataset_extract(pie_dataset_extract):
    assert pie_dataset_extract is not None


@pytest.mark.slow
def test_pie_dataset_all(pie_dataset_all, dataset_variant, split_name):
    assert pie_dataset_all is not None
    assert len(pie_dataset_all) == SPLIT_SIZES[dataset_variant][split_name]


@pytest.fixture(scope="module")
def hf_example_extract(hf_dataset_extract) -> dict:
    return list(hf_dataset_extract)[0]


@pytest.fixture(scope="module")
def pie_example_extract(pie_dataset_extract) -> DOCUMENT_TYPE:
    return list(pie_dataset_extract)[0]


def test_hf_example_extract(hf_example_extract):
    assert hf_example_extract is not None
    with open(HF_EXAMPLE_FIXTURE_PATH) as f:
        expected = json.load(f)
    assert hf_example_extract == expected


def test_pie_example_extract(pie_example_extract):
    assert pie_example_extract is not None


@pytest.fixture(scope="module")
def generate_document_kwargs_extract(hf_dataset_extract):
    return BUILDER_CLASS(config_name=DATASET_VARIANT)._generate_document_kwargs(hf_dataset_extract)


def test_generate_document_kwargs_extract(generate_document_kwargs_extract):
    assert generate_document_kwargs_extract is not None
    assert isinstance(generate_document_kwargs_extract, dict)


@pytest.fixture(scope="module")
def generate_document_kwargs_all(hf_dataset_all):
    return BUILDER_CLASS(config_name=DATASET_VARIANT)._generate_document_kwargs(hf_dataset_all)


def test_generate_document_kwargs_all(generate_document_kwargs_all):
    assert generate_document_kwargs_all is not None
    assert isinstance(generate_document_kwargs_all, dict)


@pytest.fixture(scope="module")
def generated_document_extract(hf_example_extract, generate_document_kwargs_extract):
    return BUILDER_CLASS(config_name=DATASET_VARIANT)._generate_document(
        hf_example_extract, **generate_document_kwargs_extract
    )


def test_generate_document(generated_document_extract):
    assert generated_document_extract is not None
    assert isinstance(generated_document_extract, BUILDER_CLASS.DOCUMENT_TYPE)
    # check actual annotations
    assert generated_document_extract.speakers[0].label == "Speaker#1"
    assert generated_document_extract.parts[0].label == "0"
    assert generated_document_extract.entities[0].label == "ORG"
    assert generated_document_extract.predicates[0].lemma == "memory"
    assert generated_document_extract.srl_relations[0].roles == (
        "ARG0",
        "ARGM-MNR",
        "V",
        "ARG1",
        "ARG2",
    )
    assert (
        generated_document_extract.parse_trees[0].label
        == "(TOP(SBARQ(WHNP(WHNP (WP What)  (NN kind) )(PP (IN of) (NP (NN memory) ))) (. ?) ))"
    )
    assert generated_document_extract.tokens[
        generated_document_extract.sentences[0].start : generated_document_extract.sentences[0].end
    ] == ("What", "kind", "of", "memory", "?")
    assert generated_document_extract.word_senses[0].label == "1.0"


def test_pie_example_with_generated_document_extract(
    generated_document_extract, pie_example_extract
):
    # just compare against the generated_document, we already checked that generated_document is correct
    assert pie_example_extract.id == generated_document_extract.id
    assert pie_example_extract.sentences == generated_document_extract.sentences
    assert pie_example_extract.tokens == generated_document_extract.tokens
    assert pie_example_extract.pos_tags == generated_document_extract.pos_tags
    assert pie_example_extract.entities == generated_document_extract.entities
    assert pie_example_extract.coref_mentions == generated_document_extract.coref_mentions
    assert pie_example_extract.srl_arguments == generated_document_extract.srl_arguments
    assert pie_example_extract.srl_relations == generated_document_extract.srl_relations
    assert pie_example_extract.word_senses == generated_document_extract.word_senses
    assert pie_example_extract.parts == generated_document_extract.parts

    # The annotation types for the following layers are defined in the dataset script
    # which gets copied before it is used so the types are different. Casting the
    # document converts the annotation types to the original ones so we can compare
    # the annotation layers.
    pie_example_casted = pie_example_extract.as_type(type(generated_document_extract))
    assert pie_example_casted.parse_trees == generated_document_extract.parse_trees
    assert pie_example_casted.speakers == generated_document_extract.speakers
    assert pie_example_casted.predicates == generated_document_extract.predicates
    assert pie_example_casted.coref_clusters == generated_document_extract.coref_clusters


def test_generate_example_and_back_extract(
    hf_example_extract, generated_document_extract, generate_document_kwargs_extract
):
    hf_ex_back = document_to_example(
        generated_document_extract, **generate_document_kwargs_extract
    )
    assert hf_ex_back["document_id"] == hf_example_extract["document_id"]
    for hf, gen in zip(hf_example_extract["sentences"], hf_ex_back["sentences"]):
        for key in hf.keys():
            if key == "coref_spans":
                # 'coref_spans' must be sorted before compare
                assert sorted(hf[key]) == sorted(gen[key])
            else:
                assert hf[key] == gen[key]


@pytest.mark.slow
def test_generate_example_and_back_all(
    hf_dataset_all, generate_document_kwargs_all, dataset_variant, split_name
):
    i = 0
    for hf_ex in list(hf_dataset_all):
        doc = example_to_document(hf_ex, **generate_document_kwargs_all)
        hf_ex_back = document_to_example(doc, **generate_document_kwargs_all)
        assert hf_ex_back["document_id"] == hf_ex["document_id"]
        for ex, ex_back in zip(hf_ex["sentences"], hf_ex_back["sentences"]):
            for key in ex.keys():
                # 'coref_spans' must be sorted before comparison
                if key == "coref_spans":
                    assert sorted(ex[key]) == sorted(ex_back[key])
                else:
                    assert ex[key] == ex_back[key]
        i += 1
    assert i == SPLIT_SIZES[dataset_variant][split_name]


def test_convert_to_text_document_with_labeled_spans_and_labeled_partitions_extract(
    generated_document_extract,
):
    converted_doc = convert_to_text_document_with_labeled_spans_and_labeled_partitions(
        generated_document_extract
    )
    assert converted_doc is not None
    assert isinstance(converted_doc, TextDocumentWithLabeledSpansAndLabeledPartitions)
    assert converted_doc.id == "bc/cctv/00/cctv_0001"

    # check start and end of the text
    assert converted_doc.text.startswith(
        "What kind of memory ? We respectfully invite you to watch a special "
        "edition of Across China . WW II Landmarks on the Great Earth of China : "
        "Eternal Memories of Taihang Mountain Standing tall on Taihang Mountain is "
        "the Monument to the Hundred Regiments Offensive ."
    )
    assert converted_doc.text.endswith(
        "Scholars will give you a detailed analysis on this edition of Hot Topic "
        "Analysis . On August 17 , Taiwan 's investigation department and police "
        "solved the case and announced the March 19 shooting case was closed . This "
        "case will not be prosecuted ."
    )

    assert len(converted_doc.labeled_spans) == 393
    # check the first 10 spans
    resolved_spans_10 = converted_doc.labeled_spans.resolve()[:10]
    assert resolved_spans_10 == [
        ("ORG", "Across China"),
        (
            "WORK_OF_ART",
            "WW II Landmarks on the Great Earth of China : Eternal Memories of Taihang Mountain",
        ),
        ("LOC", "Taihang Mountain"),
        ("WORK_OF_ART", "the Monument to the Hundred Regiments Offensive"),
        ("WORK_OF_ART", "the Great Wall"),
        ("CARDINAL", "three"),
        ("CARDINAL", "two"),
        ("EVENT", "The Hundred Regiments Offensive"),
        ("ORG", "the Eighth Route Army"),
        ("EVENT", "the War of Resistance against Japan"),
    ]

    assert len(converted_doc.labeled_partitions) == 235
    # check the first 10 partitions (sentences)
    resolved_partitions_10 = converted_doc.labeled_partitions.resolve()[:10]
    assert resolved_partitions_10 == [
        ("sentence", "What kind of memory ?"),
        ("sentence", "We respectfully invite you to watch a special edition of Across China ."),
        (
            "sentence",
            "WW II Landmarks on the Great Earth of China : Eternal Memories of Taihang Mountain",
        ),
        (
            "sentence",
            "Standing tall on Taihang Mountain is the Monument to the Hundred Regiments Offensive .",
        ),
        (
            "sentence",
            "It is composed of a primary stele , secondary steles , a huge round sculpture and beacon "
            "tower , and the Great Wall , among other things .",
        ),
        ("sentence", "A primary stele , three secondary steles , and two inscribed steles ."),
        (
            "sentence",
            "The Hundred Regiments Offensive was the campaign of the largest scale launched by the "
            "Eighth Route Army during the War of Resistance against Japan .",
        ),
        (
            "sentence",
            "This campaign broke through the Japanese army 's blockade to reach base areas behind enemy "
            "lines , stirring up anti-Japanese spirit throughout the nation and influencing the "
            "situation of the anti-fascist war of the people worldwide .",
        ),
        (
            "sentence",
            "This is Zhuanbi Village , Wuxiang County of Shanxi Province , where the Eighth Route "
            "Army was headquartered back then .",
        ),
        ("sentence", "On a wall outside the headquarters we found a map ."),
    ]


@pytest.mark.slow
def test_convert_to_text_document_with_labeled_spans_and_labeled_partitions_all(pie_dataset_all):
    for doc in pie_dataset_all:
        converted_doc = convert_to_text_document_with_labeled_spans_and_labeled_partitions(doc)
        assert converted_doc is not None
        assert isinstance(converted_doc, TextDocumentWithLabeledSpansAndLabeledPartitions)
