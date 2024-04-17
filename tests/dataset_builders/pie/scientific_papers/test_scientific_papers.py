import pytest
from datasets import disable_caching, load_dataset

from dataset_builders.pie.scientific_papers.scientific_papers import ScientificPapers
from pie_datasets import load_dataset as load_pie_dataset
from tests.dataset_builders.common import PIE_BASE_PATH

disable_caching()

DATASET_NAME = "scientific_papers"
BUILDER_CLASS = ScientificPapers
STREAM_SIZE = 10
DOCUMENT_TYPE = BUILDER_CLASS.DOCUMENT_TYPE
HF_DATASET_PATH = BUILDER_CLASS.BASE_DATASET_PATH
PIE_DATASET_PATH = PIE_BASE_PATH / DATASET_NAME
SPLIT = "train"


@pytest.fixture(scope="module", params=[config.name for config in BUILDER_CLASS.BUILDER_CONFIGS])
def dataset_variant(request):
    return request.param


@pytest.fixture(scope="module")
def hf_dataset(dataset_variant):
    dataset = load_dataset(
        BUILDER_CLASS.BASE_DATASET_PATH,
        revision=BUILDER_CLASS.BASE_DATASET_REVISION,
        name=dataset_variant,
        split=SPLIT,
        streaming=True,
    )
    dataset_head = dataset.take(STREAM_SIZE)
    return list(dataset_head)


@pytest.fixture(scope="module")
def pie_dataset(dataset_variant):
    dataset = load_pie_dataset(
        str(PIE_DATASET_PATH), name=dataset_variant, split=SPLIT, streaming=True
    )
    dataset_head = dataset.take(STREAM_SIZE)
    return list(dataset_head)


@pytest.fixture(scope="module")
def hf_example(hf_dataset):
    return hf_dataset[0]


@pytest.fixture(scope="module")
def pie_example(pie_dataset):
    return pie_dataset[0]


@pytest.fixture(scope="module")
def expected_output(dataset_variant):
    results = {
        "arxiv": {
            "article": "additive models @xcite provide an important family of models for semiparametric regression or class",
            "abstract": " additive models play an important role in semiparametric statistics . \n this paper gives learning r",
            "section_names": "introduction\nmain results on learning rates\ncomparison of learning rates",
        },
        "pubmed": {
            "article": "a recent systematic analysis showed that in 2011 , 314 ( 296 - 331 ) million children younger than 5",
            "abstract": " background : the present study was carried out to assess the effects of community nutrition interve",
            "section_names": "INTRODUCTION\nMATERIALS AND METHODS\nParticipants\nInstruments\nProcedure\nFirst step\nSecond step\nThird s",
        },
    }
    return results[dataset_variant]


@pytest.fixture(scope="module")
def generate_document_kwargs(hf_dataset, dataset_variant):
    return BUILDER_CLASS(config_name=dataset_variant)._generate_document_kwargs(hf_dataset) or {}


@pytest.fixture(scope="module")
def generated_document(hf_example, generate_document_kwargs, dataset_variant):
    doc = BUILDER_CLASS(config_name=dataset_variant)._generate_document(
        hf_example, **generate_document_kwargs
    )
    return doc


@pytest.fixture(scope="module")
def generated_example(generated_document, dataset_variant):
    example = BUILDER_CLASS(config_name=dataset_variant)._generate_example(generated_document)
    return example


def test_hf_dataset(hf_dataset):
    assert hf_dataset is not None


def test_pie_dataset(pie_dataset):
    assert pie_dataset is not None


def test_hf_example(hf_example, expected_output):
    assert hf_example is not None
    assert isinstance(hf_example, dict)
    assert hf_example["article"].startswith(expected_output["article"])
    assert hf_example["abstract"].startswith(expected_output["abstract"])
    assert hf_example["section_names"].startswith(expected_output["section_names"])


def test_pie_example(pie_example, expected_output):
    assert pie_example is not None
    assert pie_example.text.startswith(expected_output["article"])
    # Note that we use the string representation of the abstract and section name annotations
    assert str(pie_example.abstract[0]).startswith(expected_output["abstract"])
    str_section_names = "\n".join(
        [str(section_name) for section_name in pie_example.section_names]
    )
    assert str_section_names.startswith(expected_output["section_names"])


def test_generate_document_kwargs(hf_dataset, generate_document_kwargs):
    assert generate_document_kwargs is not None
    assert isinstance(generate_document_kwargs, dict)


def test_generate_document(generated_document, expected_output):
    assert generated_document is not None
    assert isinstance(generated_document, DOCUMENT_TYPE)
    assert generated_document.text is not None
    assert generated_document.abstract is not None
    assert generated_document.section_names is not None

    assert generated_document.text.startswith(expected_output["article"])
    assert generated_document.abstract[0].text.startswith(expected_output["abstract"])
    str_section_names = "\n".join(
        [section_name.text for section_name in generated_document.section_names]
    )
    assert str_section_names.startswith(expected_output["section_names"])


def test_generate_example(generated_example, expected_output):
    assert generated_example is not None
    assert isinstance(generated_example, dict)

    assert generated_example["article"].startswith(expected_output["article"])
    assert generated_example["abstract"].startswith(expected_output["abstract"])
    assert generated_example["section_names"].startswith(expected_output["section_names"])


def test_compare_document_and_generated_document(generated_document, pie_example):
    assert generated_document.text == pie_example.text
    assert generated_document.abstract[0].text == pie_example.abstract[0].text
    generated_section_names = [
        section_name.text for section_name in generated_document.section_names
    ]
    pie_section_names = [section_name.text for section_name in pie_example.section_names]
    assert generated_section_names == pie_section_names
