import datasets
import pytest
from pytorch_ie import Document
from pytorch_ie.documents import TextDocumentWithLabeledSpansAndBinaryRelations

from dataset_builders.pie.comagc.comagc import (
    Comagc,
    ComagcDocument,
    convert_to_text_document_with_labeled_spans_and_binary_relations,
)
from pie_datasets import load_dataset as load_pie_dataset
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


def test_example_to_document(hf_example, builder):
    doc = builder._generate_document(hf_example)

    assert doc is not None
    assert doc.pmid == hf_example["pmid"]
    assert (
        doc.sentence
        == "Thus, FGF6 is increased in PIN and prostate cancer and can promote the proliferation of the transformed "
        "prostatic epithelial cells via paracrine and autocrine mechanisms."
    )
    assert doc.cancer_type == hf_example["cancer_type"]
    assert doc.gene.resolve() == [("FGF6", "FGF6")]
    assert doc.cancer.resolve() == [("prostate cancer", "prostate cancer")]
    assert doc.cge == hf_example["CGE"]
    assert doc.ccs == hf_example["CCS"]
    assert doc.pt == hf_example["PT"]
    assert doc.ige == hf_example["IGE"]
    assert doc.expression_change_keyword1.resolve() == []
    assert doc.expression_change_keyword2.resolve() == [
        ("increased", "Positive_regulation", "increased")
    ]


@pytest.fixture(scope="module")
def builder():
    return BUILDER_CLASS()


def test_builder(builder):
    assert builder is not None
    assert builder.dataset_name == DATASET_NAME
    assert builder.document_type == ComagcDocument


def test_example_to_document_and_back(builder, hf_example):
    generated_document = builder._generate_document(hf_example)
    hf_example_back = builder._generate_example(generated_document)
    assert hf_example_back == hf_example


def test_example_to_document_and_back_all(builder, hf_dataset):
    for example in hf_dataset["train"]:
        doc = builder._generate_document(example)
        assert doc is not None
        assert isinstance(doc, BUILDER_CLASS.DOCUMENT_TYPE)
        ex_back = builder._generate_example(doc)
        assert ex_back == example


@pytest.fixture(scope="module")
def pie_dataset():
    ds = load_pie_dataset(str(PIE_DATASET_PATH))
    return ds


def test_pie_dataset(pie_dataset):
    assert pie_dataset is not None
    assert len(pie_dataset["train"]) == SPLIT_SIZES["train"]
    for doc in pie_dataset["train"]:
        assert doc is not None
        assert isinstance(doc, Document)
        cast = doc.as_type(ComagcDocument)
        assert isinstance(cast, ComagcDocument)
        doc.copy()


@pytest.fixture(scope="module", params=list(BUILDER_CLASS.DOCUMENT_CONVERTERS))
def converted_document_type(request):
    return request.param


@pytest.fixture(scope="module")
def converted_pie_dataset(pie_dataset, converted_document_type):
    pie_dataset_converted = pie_dataset.to_document_type(document_type=converted_document_type)
    return pie_dataset_converted


def test_converted_pie_dataset(converted_pie_dataset, converted_document_type):
    assert converted_pie_dataset is not None
    assert len(converted_pie_dataset["train"]) == SPLIT_SIZES["train"]
    for doc in converted_pie_dataset["train"]:
        assert doc is not None
        assert isinstance(doc, converted_document_type)
        doc.copy()


def test_converted_document_from_pie_dataset(hf_example, builder):
    doc = builder._generate_document(hf_example)
    converted_doc = convert_to_text_document_with_labeled_spans_and_binary_relations(doc)
    assert converted_doc is not None
    assert isinstance(converted_doc, TextDocumentWithLabeledSpansAndBinaryRelations)

    assert converted_doc.id == "10945637.s12"
    assert (
        converted_doc.text
        == "Thus, FGF6 is increased in PIN and prostate cancer and can promote the proliferation of the transformed prostatic epithelial cells via paracrine and autocrine mechanisms."
    )
    assert converted_doc.labeled_spans.resolve() == [
        ("GENE", "FGF6"),
        ("CANCER", "prostate cancer"),
    ]
    assert converted_doc.binary_relations.resolve() == [
        ("oncogene", (("GENE", "FGF6"), ("CANCER", "prostate cancer")))
    ]
    assert converted_doc.metadata == {
        "CCS": "normalTOcancer",
        "CGE": "increased",
        "IGE": "unchanged",
        "PT": "causality",
        "cancer_type": "prostate",
        "expression_change_keyword_1": {"name": "\nNone\n", "pos": None, "type": None},
        "expression_change_keyword_2": {
            "name": "increased",
            "pos": [14, 22],
            "type": "Positive_regulation",
        },
    }
