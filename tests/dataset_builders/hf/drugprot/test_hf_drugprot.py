from typing import Any, Dict

import datasets
import pytest

from dataset_builders.hf.drugprot.drugprot import DrugProtDataset
from tests.dataset_builders.common import HF_BASE_PATH

DATASET_NAME = "drugprot"
HF_DATASET_PATH = str(HF_BASE_PATH / DATASET_NAME)
SPLIT_NAMES = {"train", "validation", "test_background"}
SPLIT_SIZES = {"train": 3500, "validation": 750, "test_background": 10750}


@pytest.fixture(params=[config.name for config in DrugProtDataset.BUILDER_CONFIGS], scope="module")
def dataset_variant(request) -> str:
    return request.param


@pytest.fixture(scope="module")
def hf_dataset(dataset_variant) -> datasets.DatasetDict:
    return datasets.load_dataset(HF_DATASET_PATH, name=dataset_variant)


def test_hf_dataset(hf_dataset):
    assert set(hf_dataset) == SPLIT_NAMES
    split_sizes = {split_name: len(ds) for split_name, ds in hf_dataset.items()}
    assert split_sizes == SPLIT_SIZES


@pytest.fixture(scope="module", params=list(SPLIT_NAMES))
def split(request) -> str:
    return request.param


@pytest.fixture(scope="module")
def hf_example(hf_dataset) -> Dict[str, Any]:
    return hf_dataset["train"][0]


def test_hf_example(hf_example, dataset_variant):
    if dataset_variant == "drugprot_source":
        assert hf_example == {
            "document_id": "17512723",
            "title": "RDH12, a retinol dehydrogenase causing Leber's congenital amaurosis, is also involved in steroid metabolism.",
            "abstract": "Three retinol dehydrogenases (RDHs) were tested for steroid converting abilities: human and murine RDH 12 and human RDH13. RDH12 is involved in retinal degeneration in Leber's congenital amaurosis (LCA). We show that murine Rdh12 and human RDH13 do not reveal activity towards the checked steroids, but that human type 12 RDH reduces dihydrotestosterone to androstanediol, and is thus also involved in steroid metabolism. Furthermore, we analyzed both expression and subcellular localization of these enzymes.",
            "text": "RDH12, a retinol dehydrogenase causing Leber's congenital amaurosis, is also involved in steroid metabolism. Three retinol dehydrogenases (RDHs) were tested for steroid converting abilities: human and murine RDH 12 and human RDH13. RDH12 is involved in retinal degeneration in Leber's congenital amaurosis (LCA). We show that murine Rdh12 and human RDH13 do not reveal activity towards the checked steroids, but that human type 12 RDH reduces dihydrotestosterone to androstanediol, and is thus also involved in steroid metabolism. Furthermore, we analyzed both expression and subcellular localization of these enzymes.",
            "entities": [
                {
                    "id": "17512723_T1",
                    "type": "CHEMICAL",
                    "text": "androstanediol",
                    "offset": [466, 480],
                },
                {
                    "id": "17512723_T2",
                    "type": "CHEMICAL",
                    "text": "retinol",
                    "offset": [115, 122],
                },
                {
                    "id": "17512723_T3",
                    "type": "CHEMICAL",
                    "text": "retinol",
                    "offset": [9, 16],
                },
                {
                    "id": "17512723_T4",
                    "type": "GENE-Y",
                    "text": "human RDH13",
                    "offset": [219, 230],
                },
                {
                    "id": "17512723_T5",
                    "type": "GENE-Y",
                    "text": "RDH12",
                    "offset": [232, 237],
                },
                {
                    "id": "17512723_T6",
                    "type": "GENE-Y",
                    "text": "murine Rdh12",
                    "offset": [326, 338],
                },
                {
                    "id": "17512723_T7",
                    "type": "GENE-Y",
                    "text": "human RDH13",
                    "offset": [343, 354],
                },
                {
                    "id": "17512723_T8",
                    "type": "GENE-N",
                    "text": "RDHs",
                    "offset": [139, 143],
                },
                {
                    "id": "17512723_T9",
                    "type": "GENE-Y",
                    "text": "human type 12 RDH",
                    "offset": [417, 434],
                },
                {
                    "id": "17512723_T10",
                    "type": "GENE-N",
                    "text": "retinol dehydrogenases",
                    "offset": [115, 137],
                },
                {
                    "id": "17512723_T11",
                    "type": "GENE-N",
                    "text": "human and murine RDH 12",
                    "offset": [191, 214],
                },
                {
                    "id": "17512723_T12",
                    "type": "GENE-Y",
                    "text": "RDH12",
                    "offset": [0, 5],
                },
                {
                    "id": "17512723_T13",
                    "type": "GENE-N",
                    "text": "retinol dehydrogenase",
                    "offset": [9, 30],
                },
            ],
            "relations": [
                {
                    "id": "17512723_0",
                    "type": "PRODUCT-OF",
                    "arg1_id": "17512723_T1",
                    "arg2_id": "17512723_T9",
                }
            ],
        }
    elif dataset_variant == "drugprot_bigbio_kb":
        assert hf_example == {
            "id": "17512723",
            "document_id": "17512723",
            "passages": [
                {
                    "id": "17512723_title",
                    "type": "title",
                    "text": [
                        "RDH12, a retinol dehydrogenase causing Leber's congenital amaurosis, is also involved in steroid metabolism."
                    ],
                    "offsets": [[0, 108]],
                },
                {
                    "id": "17512723_abstract",
                    "type": "abstract",
                    "text": [
                        "Three retinol dehydrogenases (RDHs) were tested for steroid converting abilities: human and murine RDH 12 and human RDH13. RDH12 is involved in retinal degeneration in Leber's congenital amaurosis (LCA). We show that murine Rdh12 and human RDH13 do not reveal activity towards the checked steroids, but that human type 12 RDH reduces dihydrotestosterone to androstanediol, and is thus also involved in steroid metabolism. Furthermore, we analyzed both expression and subcellular localization of these enzymes."
                    ],
                    "offsets": [[109, 618]],
                },
            ],
            "entities": [
                {
                    "id": "17512723_T1",
                    "type": "CHEMICAL",
                    "text": ["androstanediol"],
                    "offsets": [[466, 480]],
                    "normalized": [],
                },
                {
                    "id": "17512723_T2",
                    "type": "CHEMICAL",
                    "text": ["retinol"],
                    "offsets": [[115, 122]],
                    "normalized": [],
                },
                {
                    "id": "17512723_T3",
                    "type": "CHEMICAL",
                    "text": ["retinol"],
                    "offsets": [[9, 16]],
                    "normalized": [],
                },
                {
                    "id": "17512723_T4",
                    "type": "GENE-Y",
                    "text": ["human RDH13"],
                    "offsets": [[219, 230]],
                    "normalized": [],
                },
                {
                    "id": "17512723_T5",
                    "type": "GENE-Y",
                    "text": ["RDH12"],
                    "offsets": [[232, 237]],
                    "normalized": [],
                },
                {
                    "id": "17512723_T6",
                    "type": "GENE-Y",
                    "text": ["murine Rdh12"],
                    "offsets": [[326, 338]],
                    "normalized": [],
                },
                {
                    "id": "17512723_T7",
                    "type": "GENE-Y",
                    "text": ["human RDH13"],
                    "offsets": [[343, 354]],
                    "normalized": [],
                },
                {
                    "id": "17512723_T8",
                    "type": "GENE-N",
                    "text": ["RDHs"],
                    "offsets": [[139, 143]],
                    "normalized": [],
                },
                {
                    "id": "17512723_T9",
                    "type": "GENE-Y",
                    "text": ["human type 12 RDH"],
                    "offsets": [[417, 434]],
                    "normalized": [],
                },
                {
                    "id": "17512723_T10",
                    "type": "GENE-N",
                    "text": ["retinol dehydrogenases"],
                    "offsets": [[115, 137]],
                    "normalized": [],
                },
                {
                    "id": "17512723_T11",
                    "type": "GENE-N",
                    "text": ["human and murine RDH 12"],
                    "offsets": [[191, 214]],
                    "normalized": [],
                },
                {
                    "id": "17512723_T12",
                    "type": "GENE-Y",
                    "text": ["RDH12"],
                    "offsets": [[0, 5]],
                    "normalized": [],
                },
                {
                    "id": "17512723_T13",
                    "type": "GENE-N",
                    "text": ["retinol dehydrogenase"],
                    "offsets": [[9, 30]],
                    "normalized": [],
                },
            ],
            "events": [],
            "coreferences": [],
            "relations": [
                {
                    "id": "17512723_0",
                    "type": "PRODUCT-OF",
                    "arg1_id": "17512723_T1",
                    "arg2_id": "17512723_T9",
                    "normalized": [],
                }
            ],
        }
    else:
        raise ValueError(f"Unknown dataset variant: {dataset_variant}")


def test_hf_example_for_every_split(hf_dataset, dataset_variant, split):
    # covers both dataset variants
    example = hf_dataset[split][0]
    if split == "train" and dataset_variant == "drugprot_bigbio_kb":
        assert example["document_id"] == "17512723"
        assert len(example["entities"]) == 13
        assert len(example["relations"]) == 1
    elif split == "validation":
        assert example["document_id"] == "17651117"
        assert len(example["entities"]) == 18
        assert len(example["relations"]) == 0
    elif split == "test_background":
        assert example["document_id"] == "32733640"
        assert len(example["entities"]) == 37
        assert len(example["relations"]) == 0


def test_hf_dataset_all(hf_dataset, split):
    # covers both dataset variants
    for example in hf_dataset[split]:
        assert example["document_id"] is not None
        assert example["entities"] is not None
        if split != "test_background":
            assert example["relations"] is not None
        else:
            assert example["relations"] == []