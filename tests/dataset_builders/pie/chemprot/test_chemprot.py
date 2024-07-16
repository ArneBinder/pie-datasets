import datasets
import pytest

from dataset_builders.pie.chemprot.chemprot import Chemprot
from tests.dataset_builders.common import PIE_BASE_PATH

datasets.disable_caching()

DATASET_NAME = "chemprot"
BUILDER_CLASS = Chemprot
PIE_DATASET_PATH = PIE_BASE_PATH / DATASET_NAME
HF_DATASET_PATH = Chemprot.BASE_DATASET_PATH
SPLIT_SIZES = {"sample": 50, "test": 800, "train": 1020, "validation": 612}


@pytest.fixture(scope="module", params=[config.name for config in Chemprot.BUILDER_CONFIGS])
def dataset_variant(request) -> str:
    return "chemprot_shared_task_eval_source"


@pytest.fixture(scope="module")
def hf_dataset(dataset_variant):
    return datasets.load_dataset(HF_DATASET_PATH, name=dataset_variant)


def test_hf_dataset(hf_dataset):
    assert hf_dataset is not None
    assert set(hf_dataset) == {"sample", "train", "validation", "test"}
    split_sizes = {split_name: len(ds) for split_name, ds in hf_dataset.items()}
    assert split_sizes == SPLIT_SIZES


@pytest.fixture(scope="module")
def hf_example(hf_dataset):
    return hf_dataset["train"][0]


def test_hf_example(hf_example, dataset_variant):
    assert hf_example is not None
    if dataset_variant == "chemprot_full_source":
        assert hf_example == {
            "pmid": "16357751",
            "text": "Selective costimulation modulators: a novel approach for the treatment of rheumatoid arthritis.\nT cells have a central role in the orchestration of the immune pathways that contribute to the inflammation and joint destruction characteristic of rheumatoid arthritis (RA). The requirement for a dual signal for T-cell activation and the construction of a fusion protein that prevents engagement of the costimulatory molecules required for this activation has led to a new approach to RA therapy. This approach is mechanistically distinct from other currently used therapies; it targets events early rather than late in the immune cascade, and it results in immunomodulation rather than complete immunosuppression. The fusion protein abatacept is a selective costimulation modulator that avidly binds to the CD80/CD86 ligands on an antigen-presenting cell, resulting in the inability of these ligands to engage the CD28 receptor on the T cell. Abatacept dose-dependently reduces T-cell proliferation, serum concentrations of acute-phase reactants, and other markers of inflammation, including the production of rheumatoid factor by B cells. Recent studies have provided consistent evidence that treatment with abatacept results in a rapid onset of efficacy that is maintained over the course of treatment in patients with inadequate response to methotrexate and anti-tumor necrosis factor therapies. This efficacy includes patient-centered outcomes and radiographic measurement of disease progression. Abatacept has also demonstrated a very favorable safety profile to date. This article reviews the rationale for this therapeutic approach and highlights some of the recent studies that demonstrate the benefits obtained by using abatacept. This clinical experience indicates that abatacept is a significant addition to the therapeutic armamentarium for the management of patients with RA.",
            "entities": {
                "id": ["T1", "T2", "T3", "T4", "T5"],
                "type": ["CHEMICAL", "GENE-N", "GENE-Y", "GENE-Y", "GENE-N"],
                "text": ["methotrexate", "tumor necrosis factor", "CD80", "CD86", "CD28 receptor"],
                "offsets": [[1342, 1354], [1364, 1385], [805, 809], [810, 814], [912, 925]],
            },
            "relations": {"type": [], "arg1": [], "arg2": []},
        }
    elif dataset_variant == "chemprot_bigbio_kb":
        assert hf_example == {
            "id": "0",
            "document_id": "16357751",
            "passages": [
                {
                    "id": "1",
                    "type": "title and abstract",
                    "text": [
                        "Selective costimulation modulators: a novel approach for the treatment of rheumatoid arthritis.\nT cells have a central role in the orchestration of the immune pathways that contribute to the inflammation and joint destruction characteristic of rheumatoid arthritis (RA). The requirement for a dual signal for T-cell activation and the construction of a fusion protein that prevents engagement of the costimulatory molecules required for this activation has led to a new approach to RA therapy. This approach is mechanistically distinct from other currently used therapies; it targets events early rather than late in the immune cascade, and it results in immunomodulation rather than complete immunosuppression. The fusion protein abatacept is a selective costimulation modulator that avidly binds to the CD80/CD86 ligands on an antigen-presenting cell, resulting in the inability of these ligands to engage the CD28 receptor on the T cell. Abatacept dose-dependently reduces T-cell proliferation, serum concentrations of acute-phase reactants, and other markers of inflammation, including the production of rheumatoid factor by B cells. Recent studies have provided consistent evidence that treatment with abatacept results in a rapid onset of efficacy that is maintained over the course of treatment in patients with inadequate response to methotrexate and anti-tumor necrosis factor therapies. This efficacy includes patient-centered outcomes and radiographic measurement of disease progression. Abatacept has also demonstrated a very favorable safety profile to date. This article reviews the rationale for this therapeutic approach and highlights some of the recent studies that demonstrate the benefits obtained by using abatacept. This clinical experience indicates that abatacept is a significant addition to the therapeutic armamentarium for the management of patients with RA."
                    ],
                    "offsets": [[0, 1886]],
                }
            ],
            "entities": [
                {
                    "id": "2",
                    "type": "CHEMICAL",
                    "text": ["methotrexate"],
                    "offsets": [[1342, 1354]],
                    "normalized": [],
                },
                {
                    "id": "3",
                    "type": "GENE-N",
                    "text": ["tumor necrosis factor"],
                    "offsets": [[1364, 1385]],
                    "normalized": [],
                },
                {
                    "id": "4",
                    "type": "GENE-Y",
                    "text": ["CD80"],
                    "offsets": [[805, 809]],
                    "normalized": [],
                },
                {
                    "id": "5",
                    "type": "GENE-Y",
                    "text": ["CD86"],
                    "offsets": [[810, 814]],
                    "normalized": [],
                },
                {
                    "id": "6",
                    "type": "GENE-N",
                    "text": ["CD28 receptor"],
                    "offsets": [[912, 925]],
                    "normalized": [],
                },
            ],
            "events": [],
            "coreferences": [],
            "relations": [],
        }
    elif dataset_variant == "chemprot_shared_task_eval_source":
        assert hf_example == {
            "pmid": "16357751",
            "text": "Selective costimulation modulators: a novel approach for the treatment of rheumatoid arthritis.\nT cells have a central role in the orchestration of the immune pathways that contribute to the inflammation and joint destruction characteristic of rheumatoid arthritis (RA). The requirement for a dual signal for T-cell activation and the construction of a fusion protein that prevents engagement of the costimulatory molecules required for this activation has led to a new approach to RA therapy. This approach is mechanistically distinct from other currently used therapies; it targets events early rather than late in the immune cascade, and it results in immunomodulation rather than complete immunosuppression. The fusion protein abatacept is a selective costimulation modulator that avidly binds to the CD80/CD86 ligands on an antigen-presenting cell, resulting in the inability of these ligands to engage the CD28 receptor on the T cell. Abatacept dose-dependently reduces T-cell proliferation, serum concentrations of acute-phase reactants, and other markers of inflammation, including the production of rheumatoid factor by B cells. Recent studies have provided consistent evidence that treatment with abatacept results in a rapid onset of efficacy that is maintained over the course of treatment in patients with inadequate response to methotrexate and anti-tumor necrosis factor therapies. This efficacy includes patient-centered outcomes and radiographic measurement of disease progression. Abatacept has also demonstrated a very favorable safety profile to date. This article reviews the rationale for this therapeutic approach and highlights some of the recent studies that demonstrate the benefits obtained by using abatacept. This clinical experience indicates that abatacept is a significant addition to the therapeutic armamentarium for the management of patients with RA.",
            "entities": {
                "id": ["T1", "T2", "T3", "T4", "T5"],
                "type": ["CHEMICAL", "GENE-N", "GENE-Y", "GENE-Y", "GENE-N"],
                "text": ["methotrexate", "tumor necrosis factor", "CD80", "CD86", "CD28 receptor"],
                "offsets": [[1342, 1354], [1364, 1385], [805, 809], [810, 814], [912, 925]],
            },
            "relations": {"type": [], "arg1": [], "arg2": []},
        }
    else:
        raise ValueError(f"Unknown dataset variant: {dataset_variant}")
