import pytest
from datasets import disable_caching, load_dataset

from tests.dataset_builders.common import PIE_BASE_PATH

DATASET_NAME = "tbga"

HF_DATASET_PATH = "DFKI-SLT/tbga"
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
