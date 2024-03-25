import datasets
import pytest
from pytorch_ie.core import Document
from pytorch_ie.documents import TextDocumentWithLabeledSpansAndBinaryRelations

from dataset_builders.pie.drugprot.drugprot import Drugprot
from pie_datasets import DatasetDict
from tests.dataset_builders.common import PIE_BASE_PATH

DATASET_NAME = "drugprot"
PIE_DATASET_PATH = PIE_BASE_PATH / DATASET_NAME
HF_DATASET_PATH = Drugprot.BASE_DATASET_PATH
SPLIT_NAMES = {"train", "validation"}
SPLIT_SIZES = {"train": 3500, "validation": 750}


@pytest.fixture(params=[config.name for config in Drugprot.BUILDER_CONFIGS], scope="module")
def dataset_name(request):
    return request.param


@pytest.fixture(scope="module")
def hf_dataset(dataset_name):
    return datasets.load_dataset(HF_DATASET_PATH, name=dataset_name)


def test_hf_dataset(hf_dataset):
    assert set(hf_dataset) == SPLIT_NAMES
    split_sizes = {split_name: len(ds) for split_name, ds in hf_dataset.items()}
    assert split_sizes == SPLIT_SIZES


@pytest.fixture(scope="module")
def hf_example(hf_dataset):
    return hf_dataset["train"][0]


def test_hf_example(hf_example, dataset_name):
    if dataset_name == "drugprot_source":
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
                {"id": "17512723_T2", "type": "CHEMICAL", "text": "retinol", "offset": [115, 122]},
                {"id": "17512723_T3", "type": "CHEMICAL", "text": "retinol", "offset": [9, 16]},
                {
                    "id": "17512723_T4",
                    "type": "GENE-Y",
                    "text": "human RDH13",
                    "offset": [219, 230],
                },
                {"id": "17512723_T5", "type": "GENE-Y", "text": "RDH12", "offset": [232, 237]},
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
                {"id": "17512723_T8", "type": "GENE-N", "text": "RDHs", "offset": [139, 143]},
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
                {"id": "17512723_T12", "type": "GENE-Y", "text": "RDH12", "offset": [0, 5]},
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
    elif dataset_name == "drugprot_bigbio_kb":
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
        raise ValueError(f"Unknown dataset name: {dataset_name}")


@pytest.fixture(scope="module")
def document(hf_example, hf_dataset):
    drugprot = Drugprot()
    generate_document_kwargs = drugprot._generate_document_kwargs(hf_dataset["train"])
    document = drugprot._generate_document(example=hf_example, **generate_document_kwargs)
    return document


def test_document(document, dataset_name):
    assert isinstance(document, Document)
    if dataset_name in ("drugprot_source", "drugprot_bigbio_kb"):
        assert (
            document.text
            == "RDH12, a retinol dehydrogenase causing Leber's congenital amaurosis, is also involved in steroid metabolism. Three retinol dehydrogenases (RDHs) were tested for steroid converting abilities: human and murine RDH 12 and human RDH13. RDH12 is involved in retinal degeneration in Leber's congenital amaurosis (LCA). We show that murine Rdh12 and human RDH13 do not reveal activity towards the checked steroids, but that human type 12 RDH reduces dihydrotestosterone to androstanediol, and is thus also involved in steroid metabolism. Furthermore, we analyzed both expression and subcellular localization of these enzymes."
        )
        entities = list(document.labeled_spans)
        assert len(entities) == 13
        assert str(entities[0]) == "RDH12"
        assert str(entities[1]) == "retinol"
        assert str(entities[-1]) == "androstanediol"

        relations = list(document.binary_relations)
        assert len(relations) == 1
        assert relations[0].label == "PRODUCT-OF"
        assert str(relations[0].head) == "androstanediol"
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")


@pytest.fixture(scope="module")
def pie_dataset(dataset_name):
    return DatasetDict.load_dataset(str(PIE_DATASET_PATH), name=dataset_name)


def test_pie_dataset(pie_dataset):
    assert set(pie_dataset) == SPLIT_NAMES
    split_sizes = {split_name: len(ds) for split_name, ds in pie_dataset.items()}
    assert split_sizes == SPLIT_SIZES
