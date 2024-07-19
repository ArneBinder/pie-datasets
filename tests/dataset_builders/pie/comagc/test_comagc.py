import datasets
import pytest

from dataset_builders.pie.comagc.comagc import Comagc, example_to_document
from tests.dataset_builders.common import PIE_BASE_PATH

DATASET_NAME = "comagc"
BUILDER_CLASS = Comagc
PIE_DATASET_PATH = PIE_BASE_PATH / DATASET_NAME
HF_DATASET_PATH = BUILDER_CLASS.BASE_DATASET_PATH
# Note: The dataset does only have a train split
SPLIT_NAMES = {"train"}
SPLIT_SIZES = {"train": 821}


@pytest.fixture(scope="module")
def hf_dataset():
    return datasets.load_dataset(HF_DATASET_PATH)


def test_hf_dataset(hf_dataset):
    assert hf_dataset is not None
    assert set(hf_dataset) == SPLIT_NAMES
    split_sizes = {split_name: len(ds) for split_name, ds in hf_dataset.items()}
    assert split_sizes == SPLIT_SIZES


@pytest.fixture(scope="module")
def hf_example(hf_dataset):
    return hf_dataset["train"][0]


def test_hf_example(hf_example):
    assert hf_example is not None
    assert hf_example == {
        "pmid": "10945637.s12",
        "sentence": "Thus, FGF6 is increased in PIN and prostate cancer and can promote the proliferation of the transformed prostatic epithelial cells via paracrine and autocrine mechanisms.",
        "cancer_type": "prostate",
        "gene": {"name": "FGF6", "pos": [6, 9]},
        "cancer": {"name": "prostate cancer", "pos": [35, 49]},
        "CGE": "increased",
        "CCS": "normalTOcancer",
        "PT": "causality",
        "IGE": "unchanged",
        "expression_change_keyword_1": {"name": "\nNone\n", "pos": None, "type": None},
        "expression_change_keyword_2": {
            "name": "increased",
            "pos": [14, 22],
            "type": "Positive_regulation",
        },
    }


def test_example_to_document(hf_example):
    doc = example_to_document(hf_example)

    assert doc is not None
    assert (
        doc.text
        == "Thus, FGF6 is increased in PIN and prostate cancer and can promote the proliferation of the transformed "
        "prostatic epithelial cells via paracrine and autocrine mechanisms."
    )
    assert doc.id == hf_example["pmid"]
    assert doc.metadata == {
        "cancer_type": hf_example["cancer_type"],
        "annotation": {
            "CGE": hf_example["CGE"],
            "CCS": hf_example["CCS"],
            "PT": hf_example["PT"],
            "IGE": hf_example["IGE"],
        },
        "expression_change_keywords": [
            hf_example["expression_change_keyword_1"],
            hf_example["expression_change_keyword_2"],
        ],
    }
    assert doc.entities.resolve() == [("GENE", "FGF6"), ("CANCER", "prostate cancer")]
    assert doc.relations.resolve() == [
        ("oncogene", (("GENE", "FGF6"), ("CANCER", "prostate cancer")))
    ]


@pytest.fixture(scope="module")
def builder():
    return BUILDER_CLASS()


def test_example_to_document_all(builder, hf_dataset):
    for example in hf_dataset["train"]:
        doc = builder._generate_document(example)
        assert doc is not None
        assert isinstance(doc, BUILDER_CLASS.DOCUMENT_TYPE)


def test_document_to_example(builder, hf_example):
    generated_document = builder._generate_document(hf_example)
    hf_example_back = builder._generate_example(generated_document)
    assert hf_example_back == hf_example