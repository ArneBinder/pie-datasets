import pytest
from datasets import disable_caching, load_dataset
from pie_core import Document

from dataset_builders.pie.tbga.tbga import (
    Tbga,
    TbgaDocument,
    convert_to_text_document_with_labeled_spans_and_binary_relations,
    example_to_document,
)
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
    return load_dataset(HF_DATASET_PATH, **BUILDER_CLASS.BASE_BUILDER_KWARGS_DICT[None])


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
            ("50940", "PDE11A", "PDE11A"),
            ("C0006826", "Malignant Neoplasms", "cancer"),
        ]
        assert doc.relations.resolve() == [
            ("NA", (("50940", "PDE11A", "PDE11A"), ("C0006826", "Malignant Neoplasms", "cancer")))
        ]

    elif split == "train":
        assert doc.entities.resolve() == [
            ("6347", "CCL2", "monocyte chemoattractant protein-1"),
            ("C0231221", "Asymptomatic", "asymptomatic"),
        ]
        assert doc.relations.resolve() == [
            (
                "NA",
                (
                    ("6347", "CCL2", "monocyte chemoattractant protein-1"),
                    ("C0231221", "Asymptomatic", "asymptomatic"),
                ),
            )
        ]

    elif split == "validation":
        assert doc.entities.resolve() == [
            ("51726", "DNAJB11", "ERdj3"),
            ("C0000768", "Congenital Abnormality", "malformation"),
        ]
        assert doc.relations.resolve() == [
            (
                "NA",
                (
                    ("51726", "DNAJB11", "ERdj3"),
                    ("C0000768", "Congenital Abnormality", "malformation"),
                ),
            )
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
        # cannot assert real document type "TbgaDocument" because it comes from dataset script copied to the huggingface cache folder (and not from its origin in dataset_builders/pie/tbga)
        assert isinstance(doc, Document)
        # check that (de-)serialization works
        doc.copy()


@pytest.fixture(scope="module")
def pie_dataset_fast(split) -> IterableDataset:
    return load_pie_dataset(str(PIE_DATASET_PATH), split=split, streaming=True).take(3)


def test_pie_dataset_fast(pie_dataset_fast):
    assert pie_dataset_fast is not None


@pytest.fixture(scope="module")
def document(pie_dataset_fast) -> TbgaDocument:
    return list(pie_dataset_fast)[0]


def test_document(document, generated_document):
    assert document.text == generated_document.text
    assert document.entities.resolve() == generated_document.entities.resolve()
    assert document.relations.resolve() == generated_document.relations.resolve()


@pytest.fixture(scope="module", params=list(BUILDER_CLASS.DOCUMENT_CONVERTERS))
def converted_document_type(request):
    return request.param


def test_dataset_with_converted_documents_fast(document, converted_document_type, split):
    converted_doc = convert_to_text_document_with_labeled_spans_and_binary_relations(document)
    assert isinstance(converted_doc, converted_document_type)
    converted_doc.copy()  # check that (de-)serialization works

    if split == "test":
        assert (
            converted_doc.text
            == "In addition, the combined cancer genome expression metaanalysis datasets included PDE11A among the top 1% down-regulated genes in PCa."
        )
        assert converted_doc.labeled_spans.resolve() == [
            ("ENTITY", "PDE11A"),
            ("ENTITY", "cancer"),
        ]
        assert converted_doc.binary_relations.resolve() == [
            ("NA", (("ENTITY", "PDE11A"), ("ENTITY", "cancer")))
        ]
        assert converted_doc.metadata == {
            "entity_ids": ["50940", "C0006826"],
            "entity_names": ["PDE11A", "Malignant Neoplasms"],
        }

    elif split == "train":
        assert (
            converted_doc.text
            == "A monocyte chemoattractant protein-1 gene polymorphism is associated with occult ischemia in a high-risk asymptomatic population."
        )
        assert converted_doc.labeled_spans.resolve() == [
            ("ENTITY", "monocyte chemoattractant protein-1"),
            ("ENTITY", "asymptomatic"),
        ]

        assert converted_doc.binary_relations.resolve() == [
            ("NA", (("ENTITY", "monocyte chemoattractant protein-1"), ("ENTITY", "asymptomatic")))
        ]

        assert converted_doc.metadata == {
            "entity_ids": ["6347", "C0231221"],
            "entity_names": ["CCL2", "Asymptomatic"],
        }

    elif split == "validation":
        assert (
            converted_doc.text
            == "These results suggest that the inside-out activation of the α5β1-integrin mediated by ERdj3/Prtg/Radil signaling is crucial for proper functions of R-CNCCs, and the deficiency of this pathway causes premature apoptosis of a subset of R-CNCCs and malformation of craniofacial structures."
        )
        assert converted_doc.labeled_spans.resolve() == [
            ("ENTITY", "ERdj3"),
            ("ENTITY", "malformation"),
        ]

        assert converted_doc.binary_relations.resolve() == [
            ("NA", (("ENTITY", "ERdj3"), ("ENTITY", "malformation")))
        ]

        assert converted_doc.metadata == {
            "entity_ids": ["51726", "C0000768"],
            "entity_names": ["DNAJB11", "Congenital Abnormality"],
        }

    else:
        raise ValueError(f"Unknown split variant: {split}")


@pytest.mark.slow
def test_dataset_with_converted_documents(pie_dataset, converted_document_type):
    dataset_with_converted_documents = pie_dataset.to_document_type(converted_document_type)
    assert dataset_with_converted_documents is not None
    # check documents
    for doc in dataset_with_converted_documents:
        assert isinstance(doc, converted_document_type)
        doc.copy()  # check that (de-)serialization works
