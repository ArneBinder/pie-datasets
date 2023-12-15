import pytest
from datasets import disable_caching, load_dataset

from dataset_builders.pie.scientific_papers.scientific_papers import ScientificPapers
from tests.dataset_builders.common import PIE_BASE_PATH

disable_caching()

DATASET_NAME = "scientific_papers"
BUILDER_CLASS = ScientificPapers
STREAM_SIZE = 10
DOCUMENT_TYPE = BUILDER_CLASS.DOCUMENT_TYPE
HF_DATASET_PATH = BUILDER_CLASS.BASE_DATASET_PATH
PIE_DATASET_PATH = PIE_BASE_PATH / DATASET_NAME


@pytest.fixture(scope="module", params=[config.name for config in BUILDER_CLASS.BUILDER_CONFIGS])
def dataset_variant(request):
    return request.param


@pytest.fixture(scope="module")
def hf_dataset(dataset_variant):
    dataset = load_dataset(DATASET_NAME, dataset_variant, split="train", streaming=True)
    dataset_head = dataset.take(STREAM_SIZE)
    return list(dataset_head)


@pytest.fixture(scope="module")
def hf_example(hf_dataset):
    return hf_dataset[0]


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


def test_hf_dataset(hf_dataset):
    assert hf_dataset is not None


def test_hf_example(hf_example, expected_output):
    assert hf_example is not None
    assert isinstance(hf_example, dict)
    assert hf_example["article"].startswith(expected_output["article"])
    assert hf_example["abstract"].startswith(expected_output["abstract"])
    assert hf_example["section_names"].startswith(expected_output["section_names"])


def test_generate_document_kwargs(hf_dataset, generate_document_kwargs):
    assert generate_document_kwargs is not None
    assert isinstance(generate_document_kwargs, dict)


def test_generate_document(hf_example, generate_document_kwargs, dataset_variant, expected_output):
    doc = BUILDER_CLASS(config_name=dataset_variant)._generate_document(
        hf_example, **generate_document_kwargs
    )
    assert doc is not None
    assert isinstance(doc, DOCUMENT_TYPE)
    assert doc.text is not None
    assert doc.abstract is not None
    assert doc.section_names is not None

    assert doc.text.startswith(expected_output["article"])
    assert doc.abstract[0].text.startswith(expected_output["abstract"])
    str_section_names = "\n".join([section_name.text for section_name in doc.section_names])
    assert str_section_names.startswith(expected_output["section_names"])


def test_generate_example(hf_example, generate_document_kwargs, dataset_variant, expected_output):
    doc = BUILDER_CLASS(config_name=dataset_variant)._generate_document(
        hf_example, **generate_document_kwargs
    )
    example = BUILDER_CLASS(config_name=dataset_variant)._generate_example(doc)
    assert example is not None
    assert isinstance(example, dict)

    assert example["article"].startswith(expected_output["article"])
    assert example["abstract"].startswith(expected_output["abstract"])
    assert example["section_names"].startswith(expected_output["section_names"])
