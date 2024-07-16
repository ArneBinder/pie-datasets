import pytest
import datasets


from tests.dataset_builders.common import PIE_BASE_PATH

datasets.disable_caching()

DATASET_NAME = "chemprot"
PIE_DATASET_PATH = PIE_BASE_PATH / DATASET_NAME
HF_DATASET_PATH = "bigbio/chemprot"
SPLIT_SIZES = {'sample': 50, 'test': 800, 'train': 1020, 'validation': 612}


@pytest.fixture(scope="module")
def hf_dataset():
    return datasets.load_dataset(HF_DATASET_PATH, name="chemprot_full_source")


def test_hf_dataset(hf_dataset):
    assert hf_dataset is not None
    assert set(hf_dataset) == {"sample", "train", "validation", "test"}
    split_sizes = {split_name: len(ds) for split_name, ds in hf_dataset.items()}
    assert split_sizes == SPLIT_SIZES

