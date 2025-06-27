import pytest
from datasets import disable_caching, load_dataset
from pie_core import Document

from dataset_builders.pie.biorel.biorel import (
    BioRel,
    BioRelDocument,
    convert_to_text_document_with_labeled_spans_and_binary_relations,
    example_to_document,
)
from pie_datasets import IterableDataset
from pie_datasets import load_dataset as load_pie_dataset
from tests.dataset_builders.common import PIE_BASE_PATH

disable_caching()

DATASET_NAME = "biorel"
BUILDER_CLASS = BioRel
HF_DATASET_PATH = BUILDER_CLASS.BASE_DATASET_PATH
PIE_DATASET_PATH = PIE_BASE_PATH / DATASET_NAME
SPLIT_NAMES = {"test", "train", "validation"}
SPLIT_SIZES = {"test": 114565, "train": 534277, "validation": 114506}
STREAM_SIZE = 3
EXAMPLE_INDEX = 0


@pytest.fixture(scope="module", params=[config.name for config in BUILDER_CLASS.BUILDER_CONFIGS])
def dataset_variant(request) -> str:
    return request.param


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
    return hf_dataset[split][EXAMPLE_INDEX]


def test_hf_example(hf_example, split):
    if split == "train":
        assert hf_example == {
            "text": "algal polysaccharide obtained from carrageenin protects 80 to 100 percent "
            "of chicken embryos against fatal infections with the lee strain of "
            "influenza virus .",
            "relation": "NA",
            "h": {"id": "C0032594", "name": "polysaccharide", "pos": [6, 20]},
            "t": {"id": "C0007289", "name": "carrageenin", "pos": [35, 46]},
        }
    elif split == "validation":
        assert hf_example == {
            "text": "zeranol implants did not significantly affect marketing-transit weight changes ,"
            " but increased ( p less than .05 ) daily weight gains at all time periods in both trials .",
            "relation": "NA",
            "h": {"id": "C0005911", "name": "weight changes", "pos": [64, 78]},
            "t": {"id": "C0005910", "name": "weight", "pos": [121, 127]},
        }
    elif split == "test":
        assert hf_example == {
            "text": "the opioid peptide dynorphin decreased somatic calcium-dependent action potential duration"
            " in a portion of mouse dorsal root ganglion ( drg ) neurons without altering resting membrane"
            " potential or conductance .",
            "relation": "NA",
            "h": {"id": "C0205753", "name": "opioid peptide", "pos": [4, 18]},
            "t": {"id": "C0013355", "name": "dynorphin", "pos": [19, 28]},
        }
    else:
        raise ValueError(f"Unknown dataset split: {split}")


def test_example_to_document(hf_example, split):
    doc = example_to_document(hf_example)
    if split == "train":
        assert doc.entities.resolve() == [
            ("C0032594", "polysaccharide", "polysaccharide"),
            ("C0007289", "carrageenin", "carrageenin"),
        ]
        assert doc.relations.resolve() == [
            (
                "NA",
                (
                    ("C0032594", "polysaccharide", "polysaccharide"),
                    ("C0007289", "carrageenin", "carrageenin"),
                ),
            )
        ]
    elif split == "validation":
        assert doc.entities.resolve() == [
            ("C0005911", "weight changes", "weight changes"),
            ("C0005910", "weight", "weight"),
        ]
        assert doc.relations.resolve() == [
            (
                "NA",
                (
                    ("C0005911", "weight changes", "weight changes"),
                    ("C0005910", "weight", "weight"),
                ),
            )
        ]
    elif split == "test":
        assert doc.entities.resolve() == [
            ("C0205753", "opioid peptide", "opioid peptide"),
            ("C0013355", "dynorphin", "dynorphin"),
        ]
        assert doc.relations.resolve() == [
            (
                "NA",
                (
                    ("C0205753", "opioid peptide", "opioid peptide"),
                    ("C0013355", "dynorphin", "dynorphin"),
                ),
            )
        ]
    else:
        raise ValueError(f"Unknown split variant: {split}")


@pytest.fixture(scope="module")
def builder() -> BUILDER_CLASS:
    return BUILDER_CLASS()


@pytest.fixture(scope="module")
def generated_document(builder, hf_example):
    return builder._generate_document(hf_example)


def test_builder(builder):
    assert builder is not None
    assert builder.dataset_name == DATASET_NAME
    assert builder.document_type == BioRelDocument


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
    return load_dataset(str(PIE_DATASET_PATH), split=split, streaming=True).take(STREAM_SIZE)


def test_pie_dataset_fast(pie_dataset_fast):
    assert pie_dataset_fast is not None


@pytest.fixture(scope="module")
def document(pie_dataset_fast) -> BioRelDocument:
    return list(pie_dataset_fast)[EXAMPLE_INDEX]


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
            == "the opioid peptide dynorphin decreased somatic calcium-dependent action potential duration in a portion of mouse dorsal root ganglion ( drg ) neurons without altering resting membrane potential or conductance ."
        )
        assert converted_doc.labeled_spans.resolve() == [
            ("ENTITY", "opioid peptide"),
            ("ENTITY", "dynorphin"),
        ]
        assert converted_doc.binary_relations.resolve() == [
            ("NA", (("ENTITY", "opioid peptide"), ("ENTITY", "dynorphin")))
        ]
        assert converted_doc.metadata == {
            "entity_ids": ["C0205753", "C0013355"],
            "entity_names": ["opioid peptide", "dynorphin"],
        }

    elif split == "train":
        assert (
            converted_doc.text
            == "algal polysaccharide obtained from carrageenin protects 80 to 100 percent of chicken embryos against fatal infections with the lee strain of influenza virus ."
        )
        assert converted_doc.labeled_spans.resolve() == [
            ("ENTITY", "polysaccharide"),
            ("ENTITY", "carrageenin"),
        ]
        assert converted_doc.binary_relations.resolve() == [
            ("NA", (("ENTITY", "polysaccharide"), ("ENTITY", "carrageenin")))
        ]
        assert converted_doc.metadata == {
            "entity_ids": ["C0032594", "C0007289"],
            "entity_names": ["polysaccharide", "carrageenin"],
        }

    elif split == "validation":
        assert (
            converted_doc.text
            == "zeranol implants did not significantly affect marketing-transit weight changes , but increased ( p less than .05 ) daily weight gains at all time periods in both trials ."
        )
        assert converted_doc.labeled_spans.resolve() == [
            ("ENTITY", "weight changes"),
            ("ENTITY", "weight"),
        ]
        assert converted_doc.binary_relations.resolve() == [
            ("NA", (("ENTITY", "weight changes"), ("ENTITY", "weight")))
        ]
        assert converted_doc.metadata == {
            "entity_ids": ["C0005911", "C0005910"],
            "entity_names": ["weight changes", "weight"],
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
