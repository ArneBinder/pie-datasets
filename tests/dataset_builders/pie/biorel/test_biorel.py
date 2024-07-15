import datasets
import pytest
from datasets import disable_caching, load_dataset

from dataset_builders.pie.biorel.biorel import (
    BioRel,
    BioRelDocument,
    document_to_example,
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


def test_hf_dataset(hf_dataset, dataset_variant):
    assert hf_dataset is not None
    assert dataset_variant == "biorel"
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
    document = example_to_document(hf_example)
    if split == "train":
        assert document.entities.resolve() == [
            ("C0032594", "polysaccharide", "polysaccharide"),
            ("C0007289", "carrageenin", "carrageenin"),
        ]
        assert document.relations.resolve() == [
            (
                "NA",
                (
                    ("C0032594", "polysaccharide", "polysaccharide"),
                    ("C0007289", "carrageenin", "carrageenin"),
                ),
            )
        ]
    elif split == "validation":
        assert document.entities.resolve() == [
            ("C0005911", "weight changes", "weight changes"),
            ("C0005910", "weight", "weight"),
        ]
        assert document.relations.resolve() == [
            (
                "NA",
                (
                    ("C0005911", "weight changes", "weight changes"),
                    ("C0005910", "weight", "weight"),
                ),
            )
        ]
    elif split == "test":
        assert document.entities.resolve() == [
            ("C0205753", "opioid peptide", "opioid peptide"),
            ("C0013355", "dynorphin", "dynorphin"),
        ]
        assert document.relations.resolve() == [
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


def test_builder(builder, dataset_variant):
    assert builder is not None
    assert builder.config_id == dataset_variant
    assert builder.dataset_name == DATASET_NAME
    assert builder.document_type == BioRelDocument


def test_document_to_example(generated_document, hf_example):
    hf_example_back = document_to_example(generated_document)
    assert hf_example_back == hf_example


@pytest.fixture(scope="module")
def pie_dataset(split):
    ds = load_pie_dataset(str(PIE_DATASET_PATH), split=split)
    return ds


@pytest.mark.slow
def test_pie_dataset(pie_dataset, split):
    assert pie_dataset is not None
    assert len(pie_dataset) == SPLIT_SIZES[split]


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
