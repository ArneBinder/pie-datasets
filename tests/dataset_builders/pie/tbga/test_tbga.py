import pytest
from datasets import disable_caching, load_dataset
from pytorch_ie import Document

from dataset_builders.pie.tbga.tbga import Tbga, TbgaDocument, example_to_document
from pie_datasets import IterableDataset
from pie_datasets import load_dataset as load_pie_dataset
from tests.dataset_builders.common import PIE_BASE_PATH

DATASET_NAME = "tbga"
BUILDER_CLASS = Tbga
HF_DATASET_PATH = BUILDER_CLASS.BASE_DATASET_PATH
PIE_DATASET_PATH = PIE_BASE_PATH / DATASET_NAME
SPLIT_NAMES = {"test", "train", "validation"}
SPLIT_SIZES = {"test": 20516, "train": 178264, "validation": 20193}


disable_caching()


@pytest.fixture(scope="module")
def hf_dataset():
    return load_dataset(HF_DATASET_PATH)


@pytest.fixture(scope="module", params=list(SPLIT_SIZES))
def split(request):
    return request.param


def test_hf_dataset(hf_dataset):
    assert hf_dataset is not None
    assert set(hf_dataset) == SPLIT_NAMES
    split_sizes = {split_name: len(ds) for split_name, ds in hf_dataset.items()}
    assert split_sizes == SPLIT_SIZES


@pytest.fixture(scope="module")
def hf_example(hf_dataset, split):
    return hf_dataset[split][0]


def test_hf_example(hf_example, split):
    if split == "train":
        assert hf_example == {
            "text": "A monocyte chemoattractant protein-1 gene polymorphism is associated with occult ischemia in "
            "a high-risk asymptomatic population.",
            "relation": "NA",
            "h": {"id": 6347, "name": "CCL2", "pos": [2, 34]},
            "t": {"id": "C0231221", "name": "Asymptomatic", "pos": [105, 12]},
        }
    elif split == "validation":
        assert hf_example == {
            "text": "These results suggest that the inside-out activation of the α5β1-integrin mediated by "
            "ERdj3/Prtg/Radil signaling is crucial for proper functions of R-CNCCs, and the deficiency of this pathway "
            "causes premature apoptosis of a subset of R-CNCCs and malformation of craniofacial structures.",
            "relation": "NA",
            "h": {"id": 51726, "name": "DNAJB11", "pos": [86, 5]},
            "t": {"id": "C0000768", "name": "Congenital Abnormality", "pos": [246, 12]},
        }
    elif split == "test":
        assert hf_example == {
            "text": "In addition, the combined cancer genome expression metaanalysis datasets included PDE11A among "
            "the top 1% down-regulated genes in PCa.",
            "relation": "NA",
            "h": {"id": 50940, "name": "PDE11A", "pos": [82, 6]},
            "t": {"id": "C0006826", "name": "Malignant Neoplasms", "pos": [26, 6]},
        }
    else:
        raise ValueError(f"Invalid split: {split}")


def test_example_to_document(hf_example, split):
    doc = example_to_document(hf_example)
    if split == "test":
        assert doc.entities.resolve() == [
            (50940, "PDE11A", ""),
            ("C0006826", "Malignant Neoplasms", ""),
        ]
        assert doc.relations.resolve() == [
            ("NA", ((50940, "PDE11A", ""), ("C0006826", "Malignant Neoplasms", "")))
        ]

    elif split == "train":
        assert doc.entities.resolve() == [
            (6347, "CCL2", "monocyte chemoattractant protein"),
            ("C0231221", "Asymptomatic", ""),
        ]
        assert doc.relations.resolve() == [
            (
                "NA",
                (
                    (6347, "CCL2", "monocyte chemoattractant protein"),
                    ("C0231221", "Asymptomatic", ""),
                ),
            )
        ]

    elif split == "validation":
        assert doc.entities.resolve() == [
            (51726, "DNAJB11", ""),
            ("C0000768", "Congenital Abnormality", ""),
        ]
        assert doc.relations.resolve() == [
            ("NA", ((51726, "DNAJB11", ""), ("C0000768", "Congenital Abnormality", "")))
        ]

    else:
        raise ValueError(f"Unknown split variant: {split}")


@pytest.mark.slow
def test_example_to_document_all(hf_dataset, split):
    for example in hf_dataset[split]:
        doc = example_to_document(example)
        assert doc is not None


@pytest.fixture(scope="module")
def builder() -> BUILDER_CLASS:
    return BUILDER_CLASS()


@pytest.fixture(scope="module")
def generated_document(builder, hf_example):
    return builder._generate_document(hf_example)


def test_builder(builder):
    assert builder is not None
    assert builder.dataset_name == DATASET_NAME
    assert builder.document_type == TbgaDocument


def test_document_to_example(generated_document, builder, hf_example):
    hf_example_back = builder._generate_example(generated_document)
    assert hf_example_back == hf_example


@pytest.fixture(scope="module")
def pie_dataset(split):
    ds = load_pie_dataset(str(PIE_DATASET_PATH), split=split)
    return ds


@pytest.mark.slow
def test_pie_dataset(pie_dataset, split):
    assert pie_dataset is not None
    assert len(pie_dataset) == SPLIT_SIZES[split]
    for doc in pie_dataset:
        # cannot assert real document type "BioRelDocument" (look also test_imdb.py)
        assert isinstance(doc, Document)
        # check that (de-)serialization works
        doc.copy()


@pytest.fixture(scope="module")
def pie_dataset_fast(split) -> IterableDataset:
    return load_pie_dataset(str(PIE_DATASET_PATH), split=split, streaming=True).take(3)


def test_pie_dataset_fast(pie_dataset_fast):
    assert pie_dataset_fast is not None
