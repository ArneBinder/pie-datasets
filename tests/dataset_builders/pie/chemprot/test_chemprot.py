from typing import Union

import datasets
import pytest
from pytorch_ie import Document
from pytorch_ie.documents import (
    TextDocumentWithLabeledSpansAndBinaryRelations,
    TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions,
)

from dataset_builders.pie.chemprot.chemprot import (
    Chemprot,
    ChemprotBigbioDocument,
    ChemprotDocument,
    example_to_chemprot_bigbio_doc,
    example_to_chemprot_doc,
)
from pie_datasets import load_dataset as load_pie_dataset
from tests.dataset_builders.common import PIE_BASE_PATH

datasets.disable_caching()

DATASET_NAME = "chemprot"
BUILDER_CLASS = Chemprot
PIE_DATASET_PATH = PIE_BASE_PATH / DATASET_NAME
HF_DATASET_PATH = Chemprot.BASE_DATASET_PATH
SPLIT_NAMES = {"sample", "test", "train", "validation"}
SPLIT_SIZES = {"sample": 50, "test": 800, "train": 1020, "validation": 612}


@pytest.fixture(scope="module", params=[config.name for config in Chemprot.BUILDER_CONFIGS])
def dataset_variant(request) -> str:
    return request.param


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
    return hf_dataset["sample"][0]


def test_hf_example(hf_example, dataset_variant):
    assert hf_example is not None
    if dataset_variant == "chemprot_full_source":
        assert hf_example == {
            "pmid": "10471277",
            "text": 'Probing the salmeterol binding site on the beta 2-adrenergic receptor using a novel photoaffinity ligand, [(125)I]iodoazidosalmeterol.\nSalmeterol is a long-acting beta2-adrenergic receptor (beta 2AR) agonist used clinically to treat asthma. In addition to binding at the active agonist site, it has been proposed that salmeterol also binds with very high affinity at a second site, termed the "exosite", and that this exosite contributes to the long duration of action of salmeterol. To determine the position of the phenyl ring of the aralkyloxyalkyl side chain of salmeterol in the beta 2AR binding site, we designed and synthesized the agonist photoaffinity label [(125)I]iodoazidosalmeterol ([125I]IAS). In direct adenylyl cyclase activation, in effects on adenylyl cyclase after pretreatment of intact cells, and in guinea pig tracheal relaxation assays, IAS and the parent drug salmeterol behave essentially the same. Significantly, the photoreactive azide of IAS is positioned on the phenyl ring at the end of the molecule which is thought to be involved in exosite binding. Carrier-free radioiodinated [125I]IAS was used to photolabel epitope-tagged human beta 2AR in membranes prepared from stably transfected HEK 293 cells. Labeling with [(125)I]IAS was blocked by 10 microM (-)-alprenolol and inhibited by addition of GTP gamma S, and [125I]IAS migrated at the same position on an SDS-PAGE gel as the beta 2AR labeled by the antagonist photoaffinity label [125I]iodoazidobenzylpindolol ([125I]IABP). The labeled receptor was purified on a nickel affinity column and cleaved with factor Xa protease at a specific sequence in the large loop between transmembrane segments 5 and 6, yielding two peptides. While the control antagonist photoaffinity label [125I]IABP labeled both the large N-terminal fragment [containing transmembranes (TMs) 1-5] and the smaller C-terminal fragment (containing TMs 6 and 7), essentially all of the [125I]IAS labeling was on the smaller C-terminal peptide containing TMs 6 and 7. This direct biochemical evidence demonstrates that when salmeterol binds to the receptor, its hydrophobic aryloxyalkyl tail is positioned near TM 6 and/or TM 7. A model of IAS binding to the beta 2AR is proposed.',
            "entities": {
                "id": [
                    "T1",
                    "T2",
                    "T3",
                    "T4",
                    "T5",
                    "T6",
                    "T7",
                    "T8",
                    "T9",
                    "T10",
                    "T11",
                    "T12",
                    "T13",
                    "T14",
                    "T15",
                    "T16",
                    "T17",
                    "T18",
                    "T19",
                    "T20",
                    "T21",
                    "T22",
                    "T23",
                    "T24",
                    "T25",
                    "T26",
                    "T27",
                    "T28",
                    "T29",
                    "T30",
                    "T31",
                    "T33",
                    "T34",
                    "T35",
                    "T36",
                    "T37",
                    "T38",
                    "T39",
                    "T40",
                    "T32",
                    "T41",
                    "T42",
                    "T44",
                ],
                "type": [
                    "CHEMICAL",
                    "CHEMICAL",
                    "CHEMICAL",
                    "CHEMICAL",
                    "CHEMICAL",
                    "CHEMICAL",
                    "CHEMICAL",
                    "CHEMICAL",
                    "CHEMICAL",
                    "CHEMICAL",
                    "CHEMICAL",
                    "CHEMICAL",
                    "CHEMICAL",
                    "CHEMICAL",
                    "CHEMICAL",
                    "CHEMICAL",
                    "CHEMICAL",
                    "CHEMICAL",
                    "CHEMICAL",
                    "CHEMICAL",
                    "CHEMICAL",
                    "CHEMICAL",
                    "CHEMICAL",
                    "CHEMICAL",
                    "CHEMICAL",
                    "CHEMICAL",
                    "CHEMICAL",
                    "CHEMICAL",
                    "CHEMICAL",
                    "CHEMICAL",
                    "CHEMICAL",
                    "GENE-N",
                    "GENE-N",
                    "GENE-N",
                    "GENE-N",
                    "GENE-N",
                    "GENE-N",
                    "GENE-N",
                    "GENE-N",
                    "CHEMICAL",
                    "CHEMICAL",
                    "GENE-Y",
                    "GENE-Y",
                ],
                "text": [
                    "Salmeterol",
                    "[(125)I]IAS",
                    "(-)-alprenolol",
                    "GTP",
                    "[125I]IAS",
                    "[125I]iodoazidobenzylpindolol",
                    "[125I]IABP",
                    "nickel",
                    "[125I]IABP",
                    "N",
                    "C",
                    "[125I]IAS",
                    "salmeterol",
                    "C",
                    "salmeterol",
                    "aryloxyalkyl",
                    "IAS",
                    "salmeterol",
                    "phenyl",
                    "salmeterol",
                    "[(125)I]iodoazidosalmeterol",
                    "[125I]IAS",
                    "adenylyl",
                    "adenylyl",
                    "IAS",
                    "salmeterol",
                    "azide",
                    "IAS",
                    "phenyl",
                    "[125I]IAS",
                    "[(125)I]iodoazidosalmeterol",
                    "human beta 2AR",
                    "beta 2AR",
                    "factor Xa",
                    "beta 2AR",
                    "beta2-adrenergic receptor",
                    "beta 2AR",
                    "beta 2AR",
                    "beta 2-adrenergic receptor",
                    "salmeterol",
                    "aralkyloxyalkyl",
                    "adenylyl cyclase",
                    "adenylyl cyclase",
                ],
                "offsets": [
                    [135, 145],
                    [1248, 1259],
                    [1285, 1299],
                    [1329, 1332],
                    [1346, 1355],
                    [1467, 1496],
                    [1498, 1508],
                    [1550, 1556],
                    [1762, 1772],
                    [1796, 1797],
                    [1870, 1871],
                    [1939, 1948],
                    [318, 328],
                    [1977, 1978],
                    [2076, 2086],
                    [2126, 2138],
                    [2192, 2195],
                    [472, 482],
                    [517, 523],
                    [566, 576],
                    [667, 694],
                    [696, 705],
                    [718, 726],
                    [761, 769],
                    [860, 863],
                    [884, 894],
                    [957, 962],
                    [966, 969],
                    [991, 997],
                    [1110, 1119],
                    [106, 133],
                    [1158, 1172],
                    [1412, 1420],
                    [1590, 1599],
                    [2211, 2219],
                    [163, 188],
                    [584, 592],
                    [190, 198],
                    [43, 69],
                    [12, 22],
                    [536, 551],
                    [718, 734],
                    [761, 777],
                ],
            },
            "relations": {
                "type": [
                    "CPR:2",
                    "CPR:2",
                    "CPR:2",
                    "CPR:5",
                    "CPR:5",
                    "CPR:2",
                    "CPR:2",
                    "CPR:2",
                    "CPR:5",
                    "CPR:5",
                    "CPR:3",
                    "CPR:3",
                    "CPR:3",
                    "CPR:3",
                    "CPR:6",
                    "CPR:6",
                    "CPR:2",
                    "CPR:2",
                    "CPR:2",
                    "CPR:2",
                ],
                "arg1": [
                    "T30",
                    "T31",
                    "T32",
                    "T1",
                    "T1",
                    "T20",
                    "T41",
                    "T19",
                    "T21",
                    "T22",
                    "T25",
                    "T26",
                    "T25",
                    "T26",
                    "T6",
                    "T7",
                    "T5",
                    "T2",
                    "T17",
                    "T3",
                ],
                "arg2": [
                    "T33",
                    "T40",
                    "T40",
                    "T37",
                    "T39",
                    "T38",
                    "T38",
                    "T38",
                    "T38",
                    "T38",
                    "T42",
                    "T42",
                    "T44",
                    "T44",
                    "T34",
                    "T34",
                    "T34",
                    "T34",
                    "T36",
                    "T34",
                ],
            },
        }
    elif dataset_variant == "chemprot_bigbio_kb":
        assert hf_example == {
            "id": "0",
            "document_id": "10471277",
            "passages": [
                {
                    "id": "1",
                    "type": "title and abstract",
                    "text": [
                        'Probing the salmeterol binding site on the beta 2-adrenergic receptor using a novel photoaffinity ligand, [(125)I]iodoazidosalmeterol.\nSalmeterol is a long-acting beta2-adrenergic receptor (beta 2AR) agonist used clinically to treat asthma. In addition to binding at the active agonist site, it has been proposed that salmeterol also binds with very high affinity at a second site, termed the "exosite", and that this exosite contributes to the long duration of action of salmeterol. To determine the position of the phenyl ring of the aralkyloxyalkyl side chain of salmeterol in the beta 2AR binding site, we designed and synthesized the agonist photoaffinity label [(125)I]iodoazidosalmeterol ([125I]IAS). In direct adenylyl cyclase activation, in effects on adenylyl cyclase after pretreatment of intact cells, and in guinea pig tracheal relaxation assays, IAS and the parent drug salmeterol behave essentially the same. Significantly, the photoreactive azide of IAS is positioned on the phenyl ring at the end of the molecule which is thought to be involved in exosite binding. Carrier-free radioiodinated [125I]IAS was used to photolabel epitope-tagged human beta 2AR in membranes prepared from stably transfected HEK 293 cells. Labeling with [(125)I]IAS was blocked by 10 microM (-)-alprenolol and inhibited by addition of GTP gamma S, and [125I]IAS migrated at the same position on an SDS-PAGE gel as the beta 2AR labeled by the antagonist photoaffinity label [125I]iodoazidobenzylpindolol ([125I]IABP). The labeled receptor was purified on a nickel affinity column and cleaved with factor Xa protease at a specific sequence in the large loop between transmembrane segments 5 and 6, yielding two peptides. While the control antagonist photoaffinity label [125I]IABP labeled both the large N-terminal fragment [containing transmembranes (TMs) 1-5] and the smaller C-terminal fragment (containing TMs 6 and 7), essentially all of the [125I]IAS labeling was on the smaller C-terminal peptide containing TMs 6 and 7. This direct biochemical evidence demonstrates that when salmeterol binds to the receptor, its hydrophobic aryloxyalkyl tail is positioned near TM 6 and/or TM 7. A model of IAS binding to the beta 2AR is proposed.'
                    ],
                    "offsets": [[0, 2232]],
                }
            ],
            "entities": [
                {
                    "id": "2",
                    "type": "CHEMICAL",
                    "text": ["Salmeterol"],
                    "offsets": [[135, 145]],
                    "normalized": [],
                },
                {
                    "id": "3",
                    "type": "CHEMICAL",
                    "text": ["[(125)I]IAS"],
                    "offsets": [[1248, 1259]],
                    "normalized": [],
                },
                {
                    "id": "4",
                    "type": "CHEMICAL",
                    "text": ["(-)-alprenolol"],
                    "offsets": [[1285, 1299]],
                    "normalized": [],
                },
                {
                    "id": "5",
                    "type": "CHEMICAL",
                    "text": ["GTP"],
                    "offsets": [[1329, 1332]],
                    "normalized": [],
                },
                {
                    "id": "6",
                    "type": "CHEMICAL",
                    "text": ["[125I]IAS"],
                    "offsets": [[1346, 1355]],
                    "normalized": [],
                },
                {
                    "id": "7",
                    "type": "CHEMICAL",
                    "text": ["[125I]iodoazidobenzylpindolol"],
                    "offsets": [[1467, 1496]],
                    "normalized": [],
                },
                {
                    "id": "8",
                    "type": "CHEMICAL",
                    "text": ["[125I]IABP"],
                    "offsets": [[1498, 1508]],
                    "normalized": [],
                },
                {
                    "id": "9",
                    "type": "CHEMICAL",
                    "text": ["nickel"],
                    "offsets": [[1550, 1556]],
                    "normalized": [],
                },
                {
                    "id": "10",
                    "type": "CHEMICAL",
                    "text": ["[125I]IABP"],
                    "offsets": [[1762, 1772]],
                    "normalized": [],
                },
                {
                    "id": "11",
                    "type": "CHEMICAL",
                    "text": ["N"],
                    "offsets": [[1796, 1797]],
                    "normalized": [],
                },
                {
                    "id": "12",
                    "type": "CHEMICAL",
                    "text": ["C"],
                    "offsets": [[1870, 1871]],
                    "normalized": [],
                },
                {
                    "id": "13",
                    "type": "CHEMICAL",
                    "text": ["[125I]IAS"],
                    "offsets": [[1939, 1948]],
                    "normalized": [],
                },
                {
                    "id": "14",
                    "type": "CHEMICAL",
                    "text": ["salmeterol"],
                    "offsets": [[318, 328]],
                    "normalized": [],
                },
                {
                    "id": "15",
                    "type": "CHEMICAL",
                    "text": ["C"],
                    "offsets": [[1977, 1978]],
                    "normalized": [],
                },
                {
                    "id": "16",
                    "type": "CHEMICAL",
                    "text": ["salmeterol"],
                    "offsets": [[2076, 2086]],
                    "normalized": [],
                },
                {
                    "id": "17",
                    "type": "CHEMICAL",
                    "text": ["aryloxyalkyl"],
                    "offsets": [[2126, 2138]],
                    "normalized": [],
                },
                {
                    "id": "18",
                    "type": "CHEMICAL",
                    "text": ["IAS"],
                    "offsets": [[2192, 2195]],
                    "normalized": [],
                },
                {
                    "id": "19",
                    "type": "CHEMICAL",
                    "text": ["salmeterol"],
                    "offsets": [[472, 482]],
                    "normalized": [],
                },
                {
                    "id": "20",
                    "type": "CHEMICAL",
                    "text": ["phenyl"],
                    "offsets": [[517, 523]],
                    "normalized": [],
                },
                {
                    "id": "21",
                    "type": "CHEMICAL",
                    "text": ["salmeterol"],
                    "offsets": [[566, 576]],
                    "normalized": [],
                },
                {
                    "id": "22",
                    "type": "CHEMICAL",
                    "text": ["[(125)I]iodoazidosalmeterol"],
                    "offsets": [[667, 694]],
                    "normalized": [],
                },
                {
                    "id": "23",
                    "type": "CHEMICAL",
                    "text": ["[125I]IAS"],
                    "offsets": [[696, 705]],
                    "normalized": [],
                },
                {
                    "id": "24",
                    "type": "CHEMICAL",
                    "text": ["adenylyl"],
                    "offsets": [[718, 726]],
                    "normalized": [],
                },
                {
                    "id": "25",
                    "type": "CHEMICAL",
                    "text": ["adenylyl"],
                    "offsets": [[761, 769]],
                    "normalized": [],
                },
                {
                    "id": "26",
                    "type": "CHEMICAL",
                    "text": ["IAS"],
                    "offsets": [[860, 863]],
                    "normalized": [],
                },
                {
                    "id": "27",
                    "type": "CHEMICAL",
                    "text": ["salmeterol"],
                    "offsets": [[884, 894]],
                    "normalized": [],
                },
                {
                    "id": "28",
                    "type": "CHEMICAL",
                    "text": ["azide"],
                    "offsets": [[957, 962]],
                    "normalized": [],
                },
                {
                    "id": "29",
                    "type": "CHEMICAL",
                    "text": ["IAS"],
                    "offsets": [[966, 969]],
                    "normalized": [],
                },
                {
                    "id": "30",
                    "type": "CHEMICAL",
                    "text": ["phenyl"],
                    "offsets": [[991, 997]],
                    "normalized": [],
                },
                {
                    "id": "31",
                    "type": "CHEMICAL",
                    "text": ["[125I]IAS"],
                    "offsets": [[1110, 1119]],
                    "normalized": [],
                },
                {
                    "id": "32",
                    "type": "CHEMICAL",
                    "text": ["[(125)I]iodoazidosalmeterol"],
                    "offsets": [[106, 133]],
                    "normalized": [],
                },
                {
                    "id": "33",
                    "type": "GENE-N",
                    "text": ["human beta 2AR"],
                    "offsets": [[1158, 1172]],
                    "normalized": [],
                },
                {
                    "id": "34",
                    "type": "GENE-N",
                    "text": ["beta 2AR"],
                    "offsets": [[1412, 1420]],
                    "normalized": [],
                },
                {
                    "id": "35",
                    "type": "GENE-N",
                    "text": ["factor Xa"],
                    "offsets": [[1590, 1599]],
                    "normalized": [],
                },
                {
                    "id": "36",
                    "type": "GENE-N",
                    "text": ["beta 2AR"],
                    "offsets": [[2211, 2219]],
                    "normalized": [],
                },
                {
                    "id": "37",
                    "type": "GENE-N",
                    "text": ["beta2-adrenergic receptor"],
                    "offsets": [[163, 188]],
                    "normalized": [],
                },
                {
                    "id": "38",
                    "type": "GENE-N",
                    "text": ["beta 2AR"],
                    "offsets": [[584, 592]],
                    "normalized": [],
                },
                {
                    "id": "39",
                    "type": "GENE-N",
                    "text": ["beta 2AR"],
                    "offsets": [[190, 198]],
                    "normalized": [],
                },
                {
                    "id": "40",
                    "type": "GENE-N",
                    "text": ["beta 2-adrenergic receptor"],
                    "offsets": [[43, 69]],
                    "normalized": [],
                },
                {
                    "id": "41",
                    "type": "CHEMICAL",
                    "text": ["salmeterol"],
                    "offsets": [[12, 22]],
                    "normalized": [],
                },
                {
                    "id": "42",
                    "type": "CHEMICAL",
                    "text": ["aralkyloxyalkyl"],
                    "offsets": [[536, 551]],
                    "normalized": [],
                },
                {
                    "id": "43",
                    "type": "GENE-Y",
                    "text": ["adenylyl cyclase"],
                    "offsets": [[718, 734]],
                    "normalized": [],
                },
                {
                    "id": "44",
                    "type": "GENE-Y",
                    "text": ["adenylyl cyclase"],
                    "offsets": [[761, 777]],
                    "normalized": [],
                },
            ],
            "events": [],
            "coreferences": [],
            "relations": [
                {
                    "id": "45",
                    "type": "Regulator",
                    "arg1_id": "31",
                    "arg2_id": "33",
                    "normalized": [],
                },
                {
                    "id": "46",
                    "type": "Regulator",
                    "arg1_id": "32",
                    "arg2_id": "40",
                    "normalized": [],
                },
                {
                    "id": "47",
                    "type": "Regulator",
                    "arg1_id": "41",
                    "arg2_id": "40",
                    "normalized": [],
                },
                {"id": "48", "type": "Agonist", "arg1_id": "2", "arg2_id": "37", "normalized": []},
                {"id": "49", "type": "Agonist", "arg1_id": "2", "arg2_id": "39", "normalized": []},
                {
                    "id": "50",
                    "type": "Regulator",
                    "arg1_id": "21",
                    "arg2_id": "38",
                    "normalized": [],
                },
                {
                    "id": "51",
                    "type": "Regulator",
                    "arg1_id": "42",
                    "arg2_id": "38",
                    "normalized": [],
                },
                {
                    "id": "52",
                    "type": "Regulator",
                    "arg1_id": "20",
                    "arg2_id": "38",
                    "normalized": [],
                },
                {
                    "id": "53",
                    "type": "Agonist",
                    "arg1_id": "22",
                    "arg2_id": "38",
                    "normalized": [],
                },
                {
                    "id": "54",
                    "type": "Agonist",
                    "arg1_id": "23",
                    "arg2_id": "38",
                    "normalized": [],
                },
                {
                    "id": "55",
                    "type": "Upregulator",
                    "arg1_id": "26",
                    "arg2_id": "43",
                    "normalized": [],
                },
                {
                    "id": "56",
                    "type": "Upregulator",
                    "arg1_id": "27",
                    "arg2_id": "43",
                    "normalized": [],
                },
                {
                    "id": "57",
                    "type": "Upregulator",
                    "arg1_id": "26",
                    "arg2_id": "44",
                    "normalized": [],
                },
                {
                    "id": "58",
                    "type": "Upregulator",
                    "arg1_id": "27",
                    "arg2_id": "44",
                    "normalized": [],
                },
                {
                    "id": "59",
                    "type": "Antagonist",
                    "arg1_id": "7",
                    "arg2_id": "34",
                    "normalized": [],
                },
                {
                    "id": "60",
                    "type": "Antagonist",
                    "arg1_id": "8",
                    "arg2_id": "34",
                    "normalized": [],
                },
                {
                    "id": "61",
                    "type": "Regulator",
                    "arg1_id": "6",
                    "arg2_id": "34",
                    "normalized": [],
                },
                {
                    "id": "62",
                    "type": "Regulator",
                    "arg1_id": "3",
                    "arg2_id": "34",
                    "normalized": [],
                },
                {
                    "id": "63",
                    "type": "Regulator",
                    "arg1_id": "18",
                    "arg2_id": "36",
                    "normalized": [],
                },
                {
                    "id": "64",
                    "type": "Regulator",
                    "arg1_id": "4",
                    "arg2_id": "34",
                    "normalized": [],
                },
            ],
        }
    elif dataset_variant == "chemprot_shared_task_eval_source":
        assert hf_example == {
            "pmid": "10471277",
            "text": 'Probing the salmeterol binding site on the beta 2-adrenergic receptor using a novel photoaffinity ligand, [(125)I]iodoazidosalmeterol.\nSalmeterol is a long-acting beta2-adrenergic receptor (beta 2AR) agonist used clinically to treat asthma. In addition to binding at the active agonist site, it has been proposed that salmeterol also binds with very high affinity at a second site, termed the "exosite", and that this exosite contributes to the long duration of action of salmeterol. To determine the position of the phenyl ring of the aralkyloxyalkyl side chain of salmeterol in the beta 2AR binding site, we designed and synthesized the agonist photoaffinity label [(125)I]iodoazidosalmeterol ([125I]IAS). In direct adenylyl cyclase activation, in effects on adenylyl cyclase after pretreatment of intact cells, and in guinea pig tracheal relaxation assays, IAS and the parent drug salmeterol behave essentially the same. Significantly, the photoreactive azide of IAS is positioned on the phenyl ring at the end of the molecule which is thought to be involved in exosite binding. Carrier-free radioiodinated [125I]IAS was used to photolabel epitope-tagged human beta 2AR in membranes prepared from stably transfected HEK 293 cells. Labeling with [(125)I]IAS was blocked by 10 microM (-)-alprenolol and inhibited by addition of GTP gamma S, and [125I]IAS migrated at the same position on an SDS-PAGE gel as the beta 2AR labeled by the antagonist photoaffinity label [125I]iodoazidobenzylpindolol ([125I]IABP). The labeled receptor was purified on a nickel affinity column and cleaved with factor Xa protease at a specific sequence in the large loop between transmembrane segments 5 and 6, yielding two peptides. While the control antagonist photoaffinity label [125I]IABP labeled both the large N-terminal fragment [containing transmembranes (TMs) 1-5] and the smaller C-terminal fragment (containing TMs 6 and 7), essentially all of the [125I]IAS labeling was on the smaller C-terminal peptide containing TMs 6 and 7. This direct biochemical evidence demonstrates that when salmeterol binds to the receptor, its hydrophobic aryloxyalkyl tail is positioned near TM 6 and/or TM 7. A model of IAS binding to the beta 2AR is proposed.',
            "entities": {
                "id": [
                    "T1",
                    "T2",
                    "T3",
                    "T4",
                    "T5",
                    "T6",
                    "T7",
                    "T8",
                    "T9",
                    "T10",
                    "T11",
                    "T12",
                    "T13",
                    "T14",
                    "T15",
                    "T16",
                    "T17",
                    "T18",
                    "T19",
                    "T20",
                    "T21",
                    "T22",
                    "T23",
                    "T24",
                    "T25",
                    "T26",
                    "T27",
                    "T28",
                    "T29",
                    "T30",
                    "T31",
                    "T33",
                    "T34",
                    "T35",
                    "T36",
                    "T37",
                    "T38",
                    "T39",
                    "T40",
                    "T32",
                    "T41",
                    "T42",
                    "T44",
                ],
                "type": [
                    "CHEMICAL",
                    "CHEMICAL",
                    "CHEMICAL",
                    "CHEMICAL",
                    "CHEMICAL",
                    "CHEMICAL",
                    "CHEMICAL",
                    "CHEMICAL",
                    "CHEMICAL",
                    "CHEMICAL",
                    "CHEMICAL",
                    "CHEMICAL",
                    "CHEMICAL",
                    "CHEMICAL",
                    "CHEMICAL",
                    "CHEMICAL",
                    "CHEMICAL",
                    "CHEMICAL",
                    "CHEMICAL",
                    "CHEMICAL",
                    "CHEMICAL",
                    "CHEMICAL",
                    "CHEMICAL",
                    "CHEMICAL",
                    "CHEMICAL",
                    "CHEMICAL",
                    "CHEMICAL",
                    "CHEMICAL",
                    "CHEMICAL",
                    "CHEMICAL",
                    "CHEMICAL",
                    "GENE-N",
                    "GENE-N",
                    "GENE-N",
                    "GENE-N",
                    "GENE-N",
                    "GENE-N",
                    "GENE-N",
                    "GENE-N",
                    "CHEMICAL",
                    "CHEMICAL",
                    "GENE-Y",
                    "GENE-Y",
                ],
                "text": [
                    "Salmeterol",
                    "[(125)I]IAS",
                    "(-)-alprenolol",
                    "GTP",
                    "[125I]IAS",
                    "[125I]iodoazidobenzylpindolol",
                    "[125I]IABP",
                    "nickel",
                    "[125I]IABP",
                    "N",
                    "C",
                    "[125I]IAS",
                    "salmeterol",
                    "C",
                    "salmeterol",
                    "aryloxyalkyl",
                    "IAS",
                    "salmeterol",
                    "phenyl",
                    "salmeterol",
                    "[(125)I]iodoazidosalmeterol",
                    "[125I]IAS",
                    "adenylyl",
                    "adenylyl",
                    "IAS",
                    "salmeterol",
                    "azide",
                    "IAS",
                    "phenyl",
                    "[125I]IAS",
                    "[(125)I]iodoazidosalmeterol",
                    "human beta 2AR",
                    "beta 2AR",
                    "factor Xa",
                    "beta 2AR",
                    "beta2-adrenergic receptor",
                    "beta 2AR",
                    "beta 2AR",
                    "beta 2-adrenergic receptor",
                    "salmeterol",
                    "aralkyloxyalkyl",
                    "adenylyl cyclase",
                    "adenylyl cyclase",
                ],
                "offsets": [
                    [135, 145],
                    [1248, 1259],
                    [1285, 1299],
                    [1329, 1332],
                    [1346, 1355],
                    [1467, 1496],
                    [1498, 1508],
                    [1550, 1556],
                    [1762, 1772],
                    [1796, 1797],
                    [1870, 1871],
                    [1939, 1948],
                    [318, 328],
                    [1977, 1978],
                    [2076, 2086],
                    [2126, 2138],
                    [2192, 2195],
                    [472, 482],
                    [517, 523],
                    [566, 576],
                    [667, 694],
                    [696, 705],
                    [718, 726],
                    [761, 769],
                    [860, 863],
                    [884, 894],
                    [957, 962],
                    [966, 969],
                    [991, 997],
                    [1110, 1119],
                    [106, 133],
                    [1158, 1172],
                    [1412, 1420],
                    [1590, 1599],
                    [2211, 2219],
                    [163, 188],
                    [584, 592],
                    [190, 198],
                    [43, 69],
                    [12, 22],
                    [536, 551],
                    [718, 734],
                    [761, 777],
                ],
            },
            "relations": {
                "type": [
                    "CPR:5",
                    "CPR:5",
                    "CPR:5",
                    "CPR:5",
                    "CPR:3",
                    "CPR:3",
                    "CPR:3",
                    "CPR:3",
                    "CPR:6",
                    "CPR:6",
                ],
                "arg1": ["T1", "T1", "T21", "T22", "T25", "T26", "T25", "T26", "T6", "T7"],
                "arg2": ["T37", "T39", "T38", "T38", "T42", "T42", "T44", "T44", "T34", "T34"],
            },
        }
    else:
        raise ValueError(f"Unknown dataset variant: {dataset_variant}")


def test_example_to_document(hf_example, dataset_variant):
    assert hf_example is not None
    if dataset_variant == "chemprot_full_source":
        doc = example_to_chemprot_doc(hf_example)
        assert isinstance(doc, ChemprotDocument)
        assert doc.text == hf_example["text"]
        assert doc.entities.resolve() == [
            ("CHEMICAL", "Salmeterol"),
            ("CHEMICAL", "[(125)I]IAS"),
            ("CHEMICAL", "(-)-alprenolol"),
            ("CHEMICAL", "GTP"),
            ("CHEMICAL", "[125I]IAS"),
            ("CHEMICAL", "[125I]iodoazidobenzylpindolol"),
            ("CHEMICAL", "[125I]IABP"),
            ("CHEMICAL", "nickel"),
            ("CHEMICAL", "[125I]IABP"),
            ("CHEMICAL", "N"),
            ("CHEMICAL", "C"),
            ("CHEMICAL", "[125I]IAS"),
            ("CHEMICAL", "salmeterol"),
            ("CHEMICAL", "C"),
            ("CHEMICAL", "salmeterol"),
            ("CHEMICAL", "aryloxyalkyl"),
            ("CHEMICAL", "IAS"),
            ("CHEMICAL", "salmeterol"),
            ("CHEMICAL", "phenyl"),
            ("CHEMICAL", "salmeterol"),
            ("CHEMICAL", "[(125)I]iodoazidosalmeterol"),
            ("CHEMICAL", "[125I]IAS"),
            ("CHEMICAL", "adenylyl"),
            ("CHEMICAL", "adenylyl"),
            ("CHEMICAL", "IAS"),
            ("CHEMICAL", "salmeterol"),
            ("CHEMICAL", "azide"),
            ("CHEMICAL", "IAS"),
            ("CHEMICAL", "phenyl"),
            ("CHEMICAL", "[125I]IAS"),
            ("CHEMICAL", "[(125)I]iodoazidosalmeterol"),
            ("GENE-N", "human beta 2AR"),
            ("GENE-N", "beta 2AR"),
            ("GENE-N", "factor Xa"),
            ("GENE-N", "beta 2AR"),
            ("GENE-N", "beta2-adrenergic receptor"),
            ("GENE-N", "beta 2AR"),
            ("GENE-N", "beta 2AR"),
            ("GENE-N", "beta 2-adrenergic receptor"),
            ("CHEMICAL", "salmeterol"),
            ("CHEMICAL", "aralkyloxyalkyl"),
            ("GENE-Y", "adenylyl cyclase"),
            ("GENE-Y", "adenylyl cyclase"),
        ]
        assert doc.relations.resolve() == [
            ("CPR:2", (("CHEMICAL", "[125I]IAS"), ("GENE-N", "human beta 2AR"))),
            (
                "CPR:2",
                (
                    ("CHEMICAL", "[(125)I]iodoazidosalmeterol"),
                    ("GENE-N", "beta 2-adrenergic receptor"),
                ),
            ),
            ("CPR:2", (("CHEMICAL", "salmeterol"), ("GENE-N", "beta 2-adrenergic receptor"))),
            ("CPR:5", (("CHEMICAL", "Salmeterol"), ("GENE-N", "beta2-adrenergic receptor"))),
            ("CPR:5", (("CHEMICAL", "Salmeterol"), ("GENE-N", "beta 2AR"))),
            ("CPR:2", (("CHEMICAL", "salmeterol"), ("GENE-N", "beta 2AR"))),
            ("CPR:2", (("CHEMICAL", "aralkyloxyalkyl"), ("GENE-N", "beta 2AR"))),
            ("CPR:2", (("CHEMICAL", "phenyl"), ("GENE-N", "beta 2AR"))),
            ("CPR:5", (("CHEMICAL", "[(125)I]iodoazidosalmeterol"), ("GENE-N", "beta 2AR"))),
            ("CPR:5", (("CHEMICAL", "[125I]IAS"), ("GENE-N", "beta 2AR"))),
            ("CPR:3", (("CHEMICAL", "IAS"), ("GENE-Y", "adenylyl cyclase"))),
            ("CPR:3", (("CHEMICAL", "salmeterol"), ("GENE-Y", "adenylyl cyclase"))),
            ("CPR:3", (("CHEMICAL", "IAS"), ("GENE-Y", "adenylyl cyclase"))),
            ("CPR:3", (("CHEMICAL", "salmeterol"), ("GENE-Y", "adenylyl cyclase"))),
            ("CPR:6", (("CHEMICAL", "[125I]iodoazidobenzylpindolol"), ("GENE-N", "beta 2AR"))),
            ("CPR:6", (("CHEMICAL", "[125I]IABP"), ("GENE-N", "beta 2AR"))),
            ("CPR:2", (("CHEMICAL", "[125I]IAS"), ("GENE-N", "beta 2AR"))),
            ("CPR:2", (("CHEMICAL", "[(125)I]IAS"), ("GENE-N", "beta 2AR"))),
            ("CPR:2", (("CHEMICAL", "IAS"), ("GENE-N", "beta 2AR"))),
            ("CPR:2", (("CHEMICAL", "(-)-alprenolol"), ("GENE-N", "beta 2AR"))),
        ]

    elif dataset_variant == "chemprot_bigbio_kb":
        doc = example_to_chemprot_bigbio_doc(hf_example)
        assert isinstance(doc, ChemprotBigbioDocument)
        assert doc.passages.resolve() == [
            (
                "title and abstract",
                'Probing the salmeterol binding site on the beta 2-adrenergic receptor using a novel photoaffinity ligand, [(125)I]iodoazidosalmeterol.\nSalmeterol is a long-acting beta2-adrenergic receptor (beta 2AR) agonist used clinically to treat asthma. In addition to binding at the active agonist site, it has been proposed that salmeterol also binds with very high affinity at a second site, termed the "exosite", and that this exosite contributes to the long duration of action of salmeterol. To determine the position of the phenyl ring of the aralkyloxyalkyl side chain of salmeterol in the beta 2AR binding site, we designed and synthesized the agonist photoaffinity label [(125)I]iodoazidosalmeterol ([125I]IAS). In direct adenylyl cyclase activation, in effects on adenylyl cyclase after pretreatment of intact cells, and in guinea pig tracheal relaxation assays, IAS and the parent drug salmeterol behave essentially the same. Significantly, the photoreactive azide of IAS is positioned on the phenyl ring at the end of the molecule which is thought to be involved in exosite binding. Carrier-free radioiodinated [125I]IAS was used to photolabel epitope-tagged human beta 2AR in membranes prepared from stably transfected HEK 293 cells. Labeling with [(125)I]IAS was blocked by 10 microM (-)-alprenolol and inhibited by addition of GTP gamma S, and [125I]IAS migrated at the same position on an SDS-PAGE gel as the beta 2AR labeled by the antagonist photoaffinity label [125I]iodoazidobenzylpindolol ([125I]IABP). The labeled receptor was purified on a nickel affinity column and cleaved with factor Xa protease at a specific sequence in the large loop between transmembrane segments 5 and 6, yielding two peptides. While the control antagonist photoaffinity label [125I]IABP labeled both the large N-terminal fragment [containing transmembranes (TMs) 1-5] and the smaller C-terminal fragment (containing TMs 6 and 7), essentially all of the [125I]IAS labeling was on the smaller C-terminal peptide containing TMs 6 and 7. This direct biochemical evidence demonstrates that when salmeterol binds to the receptor, its hydrophobic aryloxyalkyl tail is positioned near TM 6 and/or TM 7. A model of IAS binding to the beta 2AR is proposed.',
            )
        ]
        assert doc.entities.resolve() == [
            ("CHEMICAL", "Salmeterol"),
            ("CHEMICAL", "[(125)I]IAS"),
            ("CHEMICAL", "(-)-alprenolol"),
            ("CHEMICAL", "GTP"),
            ("CHEMICAL", "[125I]IAS"),
            ("CHEMICAL", "[125I]iodoazidobenzylpindolol"),
            ("CHEMICAL", "[125I]IABP"),
            ("CHEMICAL", "nickel"),
            ("CHEMICAL", "[125I]IABP"),
            ("CHEMICAL", "N"),
            ("CHEMICAL", "C"),
            ("CHEMICAL", "[125I]IAS"),
            ("CHEMICAL", "salmeterol"),
            ("CHEMICAL", "C"),
            ("CHEMICAL", "salmeterol"),
            ("CHEMICAL", "aryloxyalkyl"),
            ("CHEMICAL", "IAS"),
            ("CHEMICAL", "salmeterol"),
            ("CHEMICAL", "phenyl"),
            ("CHEMICAL", "salmeterol"),
            ("CHEMICAL", "[(125)I]iodoazidosalmeterol"),
            ("CHEMICAL", "[125I]IAS"),
            ("CHEMICAL", "adenylyl"),
            ("CHEMICAL", "adenylyl"),
            ("CHEMICAL", "IAS"),
            ("CHEMICAL", "salmeterol"),
            ("CHEMICAL", "azide"),
            ("CHEMICAL", "IAS"),
            ("CHEMICAL", "phenyl"),
            ("CHEMICAL", "[125I]IAS"),
            ("CHEMICAL", "[(125)I]iodoazidosalmeterol"),
            ("GENE-N", "human beta 2AR"),
            ("GENE-N", "beta 2AR"),
            ("GENE-N", "factor Xa"),
            ("GENE-N", "beta 2AR"),
            ("GENE-N", "beta2-adrenergic receptor"),
            ("GENE-N", "beta 2AR"),
            ("GENE-N", "beta 2AR"),
            ("GENE-N", "beta 2-adrenergic receptor"),
            ("CHEMICAL", "salmeterol"),
            ("CHEMICAL", "aralkyloxyalkyl"),
            ("GENE-Y", "adenylyl cyclase"),
            ("GENE-Y", "adenylyl cyclase"),
        ]
        assert doc.relations.resolve() == [
            ("Regulator", (("CHEMICAL", "[125I]IAS"), ("GENE-N", "human beta 2AR"))),
            (
                "Regulator",
                (
                    ("CHEMICAL", "[(125)I]iodoazidosalmeterol"),
                    ("GENE-N", "beta 2-adrenergic receptor"),
                ),
            ),
            ("Regulator", (("CHEMICAL", "salmeterol"), ("GENE-N", "beta 2-adrenergic receptor"))),
            ("Agonist", (("CHEMICAL", "Salmeterol"), ("GENE-N", "beta2-adrenergic receptor"))),
            ("Agonist", (("CHEMICAL", "Salmeterol"), ("GENE-N", "beta 2AR"))),
            ("Regulator", (("CHEMICAL", "salmeterol"), ("GENE-N", "beta 2AR"))),
            ("Regulator", (("CHEMICAL", "aralkyloxyalkyl"), ("GENE-N", "beta 2AR"))),
            ("Regulator", (("CHEMICAL", "phenyl"), ("GENE-N", "beta 2AR"))),
            ("Agonist", (("CHEMICAL", "[(125)I]iodoazidosalmeterol"), ("GENE-N", "beta 2AR"))),
            ("Agonist", (("CHEMICAL", "[125I]IAS"), ("GENE-N", "beta 2AR"))),
            ("Upregulator", (("CHEMICAL", "IAS"), ("GENE-Y", "adenylyl cyclase"))),
            ("Upregulator", (("CHEMICAL", "salmeterol"), ("GENE-Y", "adenylyl cyclase"))),
            ("Upregulator", (("CHEMICAL", "IAS"), ("GENE-Y", "adenylyl cyclase"))),
            ("Upregulator", (("CHEMICAL", "salmeterol"), ("GENE-Y", "adenylyl cyclase"))),
            (
                "Antagonist",
                (("CHEMICAL", "[125I]iodoazidobenzylpindolol"), ("GENE-N", "beta 2AR")),
            ),
            ("Antagonist", (("CHEMICAL", "[125I]IABP"), ("GENE-N", "beta 2AR"))),
            ("Regulator", (("CHEMICAL", "[125I]IAS"), ("GENE-N", "beta 2AR"))),
            ("Regulator", (("CHEMICAL", "[(125)I]IAS"), ("GENE-N", "beta 2AR"))),
            ("Regulator", (("CHEMICAL", "IAS"), ("GENE-N", "beta 2AR"))),
            ("Regulator", (("CHEMICAL", "(-)-alprenolol"), ("GENE-N", "beta 2AR"))),
        ]

    elif dataset_variant == "chemprot_shared_task_eval_source":
        doc = example_to_chemprot_doc(hf_example)
        assert isinstance(doc, ChemprotDocument)
        assert doc.text == hf_example["text"]
        assert doc.entities.resolve() == [
            ("CHEMICAL", "Salmeterol"),
            ("CHEMICAL", "[(125)I]IAS"),
            ("CHEMICAL", "(-)-alprenolol"),
            ("CHEMICAL", "GTP"),
            ("CHEMICAL", "[125I]IAS"),
            ("CHEMICAL", "[125I]iodoazidobenzylpindolol"),
            ("CHEMICAL", "[125I]IABP"),
            ("CHEMICAL", "nickel"),
            ("CHEMICAL", "[125I]IABP"),
            ("CHEMICAL", "N"),
            ("CHEMICAL", "C"),
            ("CHEMICAL", "[125I]IAS"),
            ("CHEMICAL", "salmeterol"),
            ("CHEMICAL", "C"),
            ("CHEMICAL", "salmeterol"),
            ("CHEMICAL", "aryloxyalkyl"),
            ("CHEMICAL", "IAS"),
            ("CHEMICAL", "salmeterol"),
            ("CHEMICAL", "phenyl"),
            ("CHEMICAL", "salmeterol"),
            ("CHEMICAL", "[(125)I]iodoazidosalmeterol"),
            ("CHEMICAL", "[125I]IAS"),
            ("CHEMICAL", "adenylyl"),
            ("CHEMICAL", "adenylyl"),
            ("CHEMICAL", "IAS"),
            ("CHEMICAL", "salmeterol"),
            ("CHEMICAL", "azide"),
            ("CHEMICAL", "IAS"),
            ("CHEMICAL", "phenyl"),
            ("CHEMICAL", "[125I]IAS"),
            ("CHEMICAL", "[(125)I]iodoazidosalmeterol"),
            ("GENE-N", "human beta 2AR"),
            ("GENE-N", "beta 2AR"),
            ("GENE-N", "factor Xa"),
            ("GENE-N", "beta 2AR"),
            ("GENE-N", "beta2-adrenergic receptor"),
            ("GENE-N", "beta 2AR"),
            ("GENE-N", "beta 2AR"),
            ("GENE-N", "beta 2-adrenergic receptor"),
            ("CHEMICAL", "salmeterol"),
            ("CHEMICAL", "aralkyloxyalkyl"),
            ("GENE-Y", "adenylyl cyclase"),
            ("GENE-Y", "adenylyl cyclase"),
        ]
        assert doc.relations.resolve() == [
            ("CPR:5", (("CHEMICAL", "Salmeterol"), ("GENE-N", "beta2-adrenergic receptor"))),
            ("CPR:5", (("CHEMICAL", "Salmeterol"), ("GENE-N", "beta 2AR"))),
            ("CPR:5", (("CHEMICAL", "[(125)I]iodoazidosalmeterol"), ("GENE-N", "beta 2AR"))),
            ("CPR:5", (("CHEMICAL", "[125I]IAS"), ("GENE-N", "beta 2AR"))),
            ("CPR:3", (("CHEMICAL", "IAS"), ("GENE-Y", "adenylyl cyclase"))),
            ("CPR:3", (("CHEMICAL", "salmeterol"), ("GENE-Y", "adenylyl cyclase"))),
            ("CPR:3", (("CHEMICAL", "IAS"), ("GENE-Y", "adenylyl cyclase"))),
            ("CPR:3", (("CHEMICAL", "salmeterol"), ("GENE-Y", "adenylyl cyclase"))),
            ("CPR:6", (("CHEMICAL", "[125I]iodoazidobenzylpindolol"), ("GENE-N", "beta 2AR"))),
            ("CPR:6", (("CHEMICAL", "[125I]IABP"), ("GENE-N", "beta 2AR"))),
        ]
    else:
        raise ValueError(f"Unknown dataset variant: {dataset_variant}")


@pytest.fixture(scope="module")
def builder(dataset_variant) -> BUILDER_CLASS:
    return BUILDER_CLASS(config_name=dataset_variant)


@pytest.fixture(scope="module")
def generated_document(builder, hf_example) -> Union[ChemprotDocument, ChemprotBigbioDocument]:
    return builder._generate_document(hf_example)


def test_builder(builder, dataset_variant):
    assert builder is not None
    assert builder.config_id == dataset_variant
    assert builder.dataset_name == "chemprot"
    assert (
        builder.document_type == ChemprotDocument
        or builder.document_type == ChemprotBigbioDocument
    )


def test_document_to_example(generated_document, builder, hf_example):
    hf_example_back = builder._generate_example(generated_document)
    assert hf_example_back == hf_example


@pytest.fixture(scope="module")
def pie_dataset(dataset_variant):
    ds = load_pie_dataset(str(PIE_DATASET_PATH), name=dataset_variant)
    return ds


def test_pie_dataset(pie_dataset, dataset_variant):
    assert pie_dataset is not None
    assert pie_dataset.num_rows == SPLIT_SIZES
    for split in pie_dataset.data:
        for doc in pie_dataset[split]:
            assert isinstance(doc, Document)

            if (
                dataset_variant == "chemprot_full_source"
                or dataset_variant == "chemprot_shared_task_eval_source"
            ):
                cast = doc.as_type(ChemprotDocument)
                assert isinstance(cast, ChemprotDocument)
            elif dataset_variant == "chemprot_bigbio_kb":
                cast = doc.as_type(ChemprotBigbioDocument)
                assert isinstance(cast, ChemprotBigbioDocument)
            else:
                raise ValueError(f"Unknown dataset variant: {dataset_variant}")

            # test deserialization
            doc.copy()


@pytest.fixture(scope="module")
def converted_document_type(dataset_variant):
    if (
        dataset_variant == "chemprot_full_source"
        or dataset_variant == "chemprot_shared_task_eval_source"
    ):
        return TextDocumentWithLabeledSpansAndBinaryRelations
    elif dataset_variant == "chemprot_bigbio_kb":
        return TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions
    else:
        raise ValueError(f"Unknown dataset variant: {dataset_variant}")


@pytest.fixture(scope="module")
def converted_pie_dataset(pie_dataset, converted_document_type):
    pie_dataset_converted = pie_dataset.to_document_type(document_type=converted_document_type)
    return pie_dataset_converted


def test_converted_pie_dataset(converted_pie_dataset, converted_document_type):
    assert set(converted_pie_dataset) == SPLIT_NAMES
    split_sizes = {split_name: len(ds) for split_name, ds in converted_pie_dataset.items()}
    assert split_sizes == SPLIT_SIZES
    for ds in converted_pie_dataset.values():
        for doc in ds:
            assert isinstance(doc, converted_document_type)
            # check that (de-)serialization works
            doc.copy()


def test_converted_document_from_pie_dataset(
    converted_pie_dataset, converted_document_type, dataset_variant
):
    assert converted_pie_dataset is not None
    converted_doc = converted_pie_dataset["sample"][0]
    assert isinstance(converted_doc, converted_document_type)

    if dataset_variant == "chemprot_full_source":
        assert (
            converted_doc.text
            == 'Probing the salmeterol binding site on the beta 2-adrenergic receptor using a novel photoaffinity ligand, [(125)I]iodoazidosalmeterol.\nSalmeterol is a long-acting beta2-adrenergic receptor (beta 2AR) agonist used clinically to treat asthma. In addition to binding at the active agonist site, it has been proposed that salmeterol also binds with very high affinity at a second site, termed the "exosite", and that this exosite contributes to the long duration of action of salmeterol. To determine the position of the phenyl ring of the aralkyloxyalkyl side chain of salmeterol in the beta 2AR binding site, we designed and synthesized the agonist photoaffinity label [(125)I]iodoazidosalmeterol ([125I]IAS). In direct adenylyl cyclase activation, in effects on adenylyl cyclase after pretreatment of intact cells, and in guinea pig tracheal relaxation assays, IAS and the parent drug salmeterol behave essentially the same. Significantly, the photoreactive azide of IAS is positioned on the phenyl ring at the end of the molecule which is thought to be involved in exosite binding. Carrier-free radioiodinated [125I]IAS was used to photolabel epitope-tagged human beta 2AR in membranes prepared from stably transfected HEK 293 cells. Labeling with [(125)I]IAS was blocked by 10 microM (-)-alprenolol and inhibited by addition of GTP gamma S, and [125I]IAS migrated at the same position on an SDS-PAGE gel as the beta 2AR labeled by the antagonist photoaffinity label [125I]iodoazidobenzylpindolol ([125I]IABP). The labeled receptor was purified on a nickel affinity column and cleaved with factor Xa protease at a specific sequence in the large loop between transmembrane segments 5 and 6, yielding two peptides. While the control antagonist photoaffinity label [125I]IABP labeled both the large N-terminal fragment [containing transmembranes (TMs) 1-5] and the smaller C-terminal fragment (containing TMs 6 and 7), essentially all of the [125I]IAS labeling was on the smaller C-terminal peptide containing TMs 6 and 7. This direct biochemical evidence demonstrates that when salmeterol binds to the receptor, its hydrophobic aryloxyalkyl tail is positioned near TM 6 and/or TM 7. A model of IAS binding to the beta 2AR is proposed.'
        )
        assert converted_doc.labeled_spans.resolve() == [
            ("CHEMICAL", "Salmeterol"),
            ("CHEMICAL", "[(125)I]IAS"),
            ("CHEMICAL", "(-)-alprenolol"),
            ("CHEMICAL", "GTP"),
            ("CHEMICAL", "[125I]IAS"),
            ("CHEMICAL", "[125I]iodoazidobenzylpindolol"),
            ("CHEMICAL", "[125I]IABP"),
            ("CHEMICAL", "nickel"),
            ("CHEMICAL", "[125I]IABP"),
            ("CHEMICAL", "N"),
            ("CHEMICAL", "C"),
            ("CHEMICAL", "[125I]IAS"),
            ("CHEMICAL", "salmeterol"),
            ("CHEMICAL", "C"),
            ("CHEMICAL", "salmeterol"),
            ("CHEMICAL", "aryloxyalkyl"),
            ("CHEMICAL", "IAS"),
            ("CHEMICAL", "salmeterol"),
            ("CHEMICAL", "phenyl"),
            ("CHEMICAL", "salmeterol"),
            ("CHEMICAL", "[(125)I]iodoazidosalmeterol"),
            ("CHEMICAL", "[125I]IAS"),
            ("CHEMICAL", "adenylyl"),
            ("CHEMICAL", "adenylyl"),
            ("CHEMICAL", "IAS"),
            ("CHEMICAL", "salmeterol"),
            ("CHEMICAL", "azide"),
            ("CHEMICAL", "IAS"),
            ("CHEMICAL", "phenyl"),
            ("CHEMICAL", "[125I]IAS"),
            ("CHEMICAL", "[(125)I]iodoazidosalmeterol"),
            ("GENE-N", "human beta 2AR"),
            ("GENE-N", "beta 2AR"),
            ("GENE-N", "factor Xa"),
            ("GENE-N", "beta 2AR"),
            ("GENE-N", "beta2-adrenergic receptor"),
            ("GENE-N", "beta 2AR"),
            ("GENE-N", "beta 2AR"),
            ("GENE-N", "beta 2-adrenergic receptor"),
            ("CHEMICAL", "salmeterol"),
            ("CHEMICAL", "aralkyloxyalkyl"),
            ("GENE-Y", "adenylyl cyclase"),
            ("GENE-Y", "adenylyl cyclase"),
        ]
        assert converted_doc.binary_relations.resolve() == [
            ("CPR:2", (("CHEMICAL", "[125I]IAS"), ("GENE-N", "human beta 2AR"))),
            (
                "CPR:2",
                (
                    ("CHEMICAL", "[(125)I]iodoazidosalmeterol"),
                    ("GENE-N", "beta 2-adrenergic receptor"),
                ),
            ),
            ("CPR:2", (("CHEMICAL", "salmeterol"), ("GENE-N", "beta 2-adrenergic receptor"))),
            ("CPR:5", (("CHEMICAL", "Salmeterol"), ("GENE-N", "beta2-adrenergic receptor"))),
            ("CPR:5", (("CHEMICAL", "Salmeterol"), ("GENE-N", "beta 2AR"))),
            ("CPR:2", (("CHEMICAL", "salmeterol"), ("GENE-N", "beta 2AR"))),
            ("CPR:2", (("CHEMICAL", "aralkyloxyalkyl"), ("GENE-N", "beta 2AR"))),
            ("CPR:2", (("CHEMICAL", "phenyl"), ("GENE-N", "beta 2AR"))),
            ("CPR:5", (("CHEMICAL", "[(125)I]iodoazidosalmeterol"), ("GENE-N", "beta 2AR"))),
            ("CPR:5", (("CHEMICAL", "[125I]IAS"), ("GENE-N", "beta 2AR"))),
            ("CPR:3", (("CHEMICAL", "IAS"), ("GENE-Y", "adenylyl cyclase"))),
            ("CPR:3", (("CHEMICAL", "salmeterol"), ("GENE-Y", "adenylyl cyclase"))),
            ("CPR:3", (("CHEMICAL", "IAS"), ("GENE-Y", "adenylyl cyclase"))),
            ("CPR:3", (("CHEMICAL", "salmeterol"), ("GENE-Y", "adenylyl cyclase"))),
            ("CPR:6", (("CHEMICAL", "[125I]iodoazidobenzylpindolol"), ("GENE-N", "beta 2AR"))),
            ("CPR:6", (("CHEMICAL", "[125I]IABP"), ("GENE-N", "beta 2AR"))),
            ("CPR:2", (("CHEMICAL", "[125I]IAS"), ("GENE-N", "beta 2AR"))),
            ("CPR:2", (("CHEMICAL", "[(125)I]IAS"), ("GENE-N", "beta 2AR"))),
            ("CPR:2", (("CHEMICAL", "IAS"), ("GENE-N", "beta 2AR"))),
            ("CPR:2", (("CHEMICAL", "(-)-alprenolol"), ("GENE-N", "beta 2AR"))),
        ]

    elif dataset_variant == "chemprot_bigbio_kb":
        assert (
            converted_doc.text
            == 'Probing the salmeterol binding site on the beta 2-adrenergic receptor using a novel photoaffinity ligand, [(125)I]iodoazidosalmeterol.\nSalmeterol is a long-acting beta2-adrenergic receptor (beta 2AR) agonist used clinically to treat asthma. In addition to binding at the active agonist site, it has been proposed that salmeterol also binds with very high affinity at a second site, termed the "exosite", and that this exosite contributes to the long duration of action of salmeterol. To determine the position of the phenyl ring of the aralkyloxyalkyl side chain of salmeterol in the beta 2AR binding site, we designed and synthesized the agonist photoaffinity label [(125)I]iodoazidosalmeterol ([125I]IAS). In direct adenylyl cyclase activation, in effects on adenylyl cyclase after pretreatment of intact cells, and in guinea pig tracheal relaxation assays, IAS and the parent drug salmeterol behave essentially the same. Significantly, the photoreactive azide of IAS is positioned on the phenyl ring at the end of the molecule which is thought to be involved in exosite binding. Carrier-free radioiodinated [125I]IAS was used to photolabel epitope-tagged human beta 2AR in membranes prepared from stably transfected HEK 293 cells. Labeling with [(125)I]IAS was blocked by 10 microM (-)-alprenolol and inhibited by addition of GTP gamma S, and [125I]IAS migrated at the same position on an SDS-PAGE gel as the beta 2AR labeled by the antagonist photoaffinity label [125I]iodoazidobenzylpindolol ([125I]IABP). The labeled receptor was purified on a nickel affinity column and cleaved with factor Xa protease at a specific sequence in the large loop between transmembrane segments 5 and 6, yielding two peptides. While the control antagonist photoaffinity label [125I]IABP labeled both the large N-terminal fragment [containing transmembranes (TMs) 1-5] and the smaller C-terminal fragment (containing TMs 6 and 7), essentially all of the [125I]IAS labeling was on the smaller C-terminal peptide containing TMs 6 and 7. This direct biochemical evidence demonstrates that when salmeterol binds to the receptor, its hydrophobic aryloxyalkyl tail is positioned near TM 6 and/or TM 7. A model of IAS binding to the beta 2AR is proposed.'
        )
        assert converted_doc.labeled_partitions.resolve() == [
            (
                "title and abstract",
                'Probing the salmeterol binding site on the beta 2-adrenergic receptor using a novel photoaffinity ligand, [(125)I]iodoazidosalmeterol.\nSalmeterol is a long-acting beta2-adrenergic receptor (beta 2AR) agonist used clinically to treat asthma. In addition to binding at the active agonist site, it has been proposed that salmeterol also binds with very high affinity at a second site, termed the "exosite", and that this exosite contributes to the long duration of action of salmeterol. To determine the position of the phenyl ring of the aralkyloxyalkyl side chain of salmeterol in the beta 2AR binding site, we designed and synthesized the agonist photoaffinity label [(125)I]iodoazidosalmeterol ([125I]IAS). In direct adenylyl cyclase activation, in effects on adenylyl cyclase after pretreatment of intact cells, and in guinea pig tracheal relaxation assays, IAS and the parent drug salmeterol behave essentially the same. Significantly, the photoreactive azide of IAS is positioned on the phenyl ring at the end of the molecule which is thought to be involved in exosite binding. Carrier-free radioiodinated [125I]IAS was used to photolabel epitope-tagged human beta 2AR in membranes prepared from stably transfected HEK 293 cells. Labeling with [(125)I]IAS was blocked by 10 microM (-)-alprenolol and inhibited by addition of GTP gamma S, and [125I]IAS migrated at the same position on an SDS-PAGE gel as the beta 2AR labeled by the antagonist photoaffinity label [125I]iodoazidobenzylpindolol ([125I]IABP). The labeled receptor was purified on a nickel affinity column and cleaved with factor Xa protease at a specific sequence in the large loop between transmembrane segments 5 and 6, yielding two peptides. While the control antagonist photoaffinity label [125I]IABP labeled both the large N-terminal fragment [containing transmembranes (TMs) 1-5] and the smaller C-terminal fragment (containing TMs 6 and 7), essentially all of the [125I]IAS labeling was on the smaller C-terminal peptide containing TMs 6 and 7. This direct biochemical evidence demonstrates that when salmeterol binds to the receptor, its hydrophobic aryloxyalkyl tail is positioned near TM 6 and/or TM 7. A model of IAS binding to the beta 2AR is proposed.',
            )
        ]
        assert converted_doc.labeled_spans.resolve() == [
            ("CHEMICAL", "Salmeterol"),
            ("CHEMICAL", "[(125)I]IAS"),
            ("CHEMICAL", "(-)-alprenolol"),
            ("CHEMICAL", "GTP"),
            ("CHEMICAL", "[125I]IAS"),
            ("CHEMICAL", "[125I]iodoazidobenzylpindolol"),
            ("CHEMICAL", "[125I]IABP"),
            ("CHEMICAL", "nickel"),
            ("CHEMICAL", "[125I]IABP"),
            ("CHEMICAL", "N"),
            ("CHEMICAL", "C"),
            ("CHEMICAL", "[125I]IAS"),
            ("CHEMICAL", "salmeterol"),
            ("CHEMICAL", "C"),
            ("CHEMICAL", "salmeterol"),
            ("CHEMICAL", "aryloxyalkyl"),
            ("CHEMICAL", "IAS"),
            ("CHEMICAL", "salmeterol"),
            ("CHEMICAL", "phenyl"),
            ("CHEMICAL", "salmeterol"),
            ("CHEMICAL", "[(125)I]iodoazidosalmeterol"),
            ("CHEMICAL", "[125I]IAS"),
            ("CHEMICAL", "adenylyl"),
            ("CHEMICAL", "adenylyl"),
            ("CHEMICAL", "IAS"),
            ("CHEMICAL", "salmeterol"),
            ("CHEMICAL", "azide"),
            ("CHEMICAL", "IAS"),
            ("CHEMICAL", "phenyl"),
            ("CHEMICAL", "[125I]IAS"),
            ("CHEMICAL", "[(125)I]iodoazidosalmeterol"),
            ("GENE-N", "human beta 2AR"),
            ("GENE-N", "beta 2AR"),
            ("GENE-N", "factor Xa"),
            ("GENE-N", "beta 2AR"),
            ("GENE-N", "beta2-adrenergic receptor"),
            ("GENE-N", "beta 2AR"),
            ("GENE-N", "beta 2AR"),
            ("GENE-N", "beta 2-adrenergic receptor"),
            ("CHEMICAL", "salmeterol"),
            ("CHEMICAL", "aralkyloxyalkyl"),
            ("GENE-Y", "adenylyl cyclase"),
            ("GENE-Y", "adenylyl cyclase"),
        ]
        assert converted_doc.binary_relations.resolve() == [
            ("Regulator", (("CHEMICAL", "[125I]IAS"), ("GENE-N", "human beta 2AR"))),
            (
                "Regulator",
                (
                    ("CHEMICAL", "[(125)I]iodoazidosalmeterol"),
                    ("GENE-N", "beta 2-adrenergic receptor"),
                ),
            ),
            ("Regulator", (("CHEMICAL", "salmeterol"), ("GENE-N", "beta 2-adrenergic receptor"))),
            ("Agonist", (("CHEMICAL", "Salmeterol"), ("GENE-N", "beta2-adrenergic receptor"))),
            ("Agonist", (("CHEMICAL", "Salmeterol"), ("GENE-N", "beta 2AR"))),
            ("Regulator", (("CHEMICAL", "salmeterol"), ("GENE-N", "beta 2AR"))),
            ("Regulator", (("CHEMICAL", "aralkyloxyalkyl"), ("GENE-N", "beta 2AR"))),
            ("Regulator", (("CHEMICAL", "phenyl"), ("GENE-N", "beta 2AR"))),
            ("Agonist", (("CHEMICAL", "[(125)I]iodoazidosalmeterol"), ("GENE-N", "beta 2AR"))),
            ("Agonist", (("CHEMICAL", "[125I]IAS"), ("GENE-N", "beta 2AR"))),
            ("Upregulator", (("CHEMICAL", "IAS"), ("GENE-Y", "adenylyl cyclase"))),
            ("Upregulator", (("CHEMICAL", "salmeterol"), ("GENE-Y", "adenylyl cyclase"))),
            ("Upregulator", (("CHEMICAL", "IAS"), ("GENE-Y", "adenylyl cyclase"))),
            ("Upregulator", (("CHEMICAL", "salmeterol"), ("GENE-Y", "adenylyl cyclase"))),
            (
                "Antagonist",
                (("CHEMICAL", "[125I]iodoazidobenzylpindolol"), ("GENE-N", "beta 2AR")),
            ),
            ("Antagonist", (("CHEMICAL", "[125I]IABP"), ("GENE-N", "beta 2AR"))),
            ("Regulator", (("CHEMICAL", "[125I]IAS"), ("GENE-N", "beta 2AR"))),
            ("Regulator", (("CHEMICAL", "[(125)I]IAS"), ("GENE-N", "beta 2AR"))),
            ("Regulator", (("CHEMICAL", "IAS"), ("GENE-N", "beta 2AR"))),
            ("Regulator", (("CHEMICAL", "(-)-alprenolol"), ("GENE-N", "beta 2AR"))),
        ]

    elif dataset_variant == "chemprot_shared_task_eval_source":
        assert (
            converted_doc.text
            == 'Probing the salmeterol binding site on the beta 2-adrenergic receptor using a novel photoaffinity ligand, [(125)I]iodoazidosalmeterol.\nSalmeterol is a long-acting beta2-adrenergic receptor (beta 2AR) agonist used clinically to treat asthma. In addition to binding at the active agonist site, it has been proposed that salmeterol also binds with very high affinity at a second site, termed the "exosite", and that this exosite contributes to the long duration of action of salmeterol. To determine the position of the phenyl ring of the aralkyloxyalkyl side chain of salmeterol in the beta 2AR binding site, we designed and synthesized the agonist photoaffinity label [(125)I]iodoazidosalmeterol ([125I]IAS). In direct adenylyl cyclase activation, in effects on adenylyl cyclase after pretreatment of intact cells, and in guinea pig tracheal relaxation assays, IAS and the parent drug salmeterol behave essentially the same. Significantly, the photoreactive azide of IAS is positioned on the phenyl ring at the end of the molecule which is thought to be involved in exosite binding. Carrier-free radioiodinated [125I]IAS was used to photolabel epitope-tagged human beta 2AR in membranes prepared from stably transfected HEK 293 cells. Labeling with [(125)I]IAS was blocked by 10 microM (-)-alprenolol and inhibited by addition of GTP gamma S, and [125I]IAS migrated at the same position on an SDS-PAGE gel as the beta 2AR labeled by the antagonist photoaffinity label [125I]iodoazidobenzylpindolol ([125I]IABP). The labeled receptor was purified on a nickel affinity column and cleaved with factor Xa protease at a specific sequence in the large loop between transmembrane segments 5 and 6, yielding two peptides. While the control antagonist photoaffinity label [125I]IABP labeled both the large N-terminal fragment [containing transmembranes (TMs) 1-5] and the smaller C-terminal fragment (containing TMs 6 and 7), essentially all of the [125I]IAS labeling was on the smaller C-terminal peptide containing TMs 6 and 7. This direct biochemical evidence demonstrates that when salmeterol binds to the receptor, its hydrophobic aryloxyalkyl tail is positioned near TM 6 and/or TM 7. A model of IAS binding to the beta 2AR is proposed.'
        )
        assert converted_doc.labeled_spans.resolve() == [
            ("CHEMICAL", "Salmeterol"),
            ("CHEMICAL", "[(125)I]IAS"),
            ("CHEMICAL", "(-)-alprenolol"),
            ("CHEMICAL", "GTP"),
            ("CHEMICAL", "[125I]IAS"),
            ("CHEMICAL", "[125I]iodoazidobenzylpindolol"),
            ("CHEMICAL", "[125I]IABP"),
            ("CHEMICAL", "nickel"),
            ("CHEMICAL", "[125I]IABP"),
            ("CHEMICAL", "N"),
            ("CHEMICAL", "C"),
            ("CHEMICAL", "[125I]IAS"),
            ("CHEMICAL", "salmeterol"),
            ("CHEMICAL", "C"),
            ("CHEMICAL", "salmeterol"),
            ("CHEMICAL", "aryloxyalkyl"),
            ("CHEMICAL", "IAS"),
            ("CHEMICAL", "salmeterol"),
            ("CHEMICAL", "phenyl"),
            ("CHEMICAL", "salmeterol"),
            ("CHEMICAL", "[(125)I]iodoazidosalmeterol"),
            ("CHEMICAL", "[125I]IAS"),
            ("CHEMICAL", "adenylyl"),
            ("CHEMICAL", "adenylyl"),
            ("CHEMICAL", "IAS"),
            ("CHEMICAL", "salmeterol"),
            ("CHEMICAL", "azide"),
            ("CHEMICAL", "IAS"),
            ("CHEMICAL", "phenyl"),
            ("CHEMICAL", "[125I]IAS"),
            ("CHEMICAL", "[(125)I]iodoazidosalmeterol"),
            ("GENE-N", "human beta 2AR"),
            ("GENE-N", "beta 2AR"),
            ("GENE-N", "factor Xa"),
            ("GENE-N", "beta 2AR"),
            ("GENE-N", "beta2-adrenergic receptor"),
            ("GENE-N", "beta 2AR"),
            ("GENE-N", "beta 2AR"),
            ("GENE-N", "beta 2-adrenergic receptor"),
            ("CHEMICAL", "salmeterol"),
            ("CHEMICAL", "aralkyloxyalkyl"),
            ("GENE-Y", "adenylyl cyclase"),
            ("GENE-Y", "adenylyl cyclase"),
        ]
        assert converted_doc.binary_relations.resolve() == [
            ("CPR:5", (("CHEMICAL", "Salmeterol"), ("GENE-N", "beta2-adrenergic receptor"))),
            ("CPR:5", (("CHEMICAL", "Salmeterol"), ("GENE-N", "beta 2AR"))),
            ("CPR:5", (("CHEMICAL", "[(125)I]iodoazidosalmeterol"), ("GENE-N", "beta 2AR"))),
            ("CPR:5", (("CHEMICAL", "[125I]IAS"), ("GENE-N", "beta 2AR"))),
            ("CPR:3", (("CHEMICAL", "IAS"), ("GENE-Y", "adenylyl cyclase"))),
            ("CPR:3", (("CHEMICAL", "salmeterol"), ("GENE-Y", "adenylyl cyclase"))),
            ("CPR:3", (("CHEMICAL", "IAS"), ("GENE-Y", "adenylyl cyclase"))),
            ("CPR:3", (("CHEMICAL", "salmeterol"), ("GENE-Y", "adenylyl cyclase"))),
            ("CPR:6", (("CHEMICAL", "[125I]iodoazidobenzylpindolol"), ("GENE-N", "beta 2AR"))),
            ("CPR:6", (("CHEMICAL", "[125I]IABP"), ("GENE-N", "beta 2AR"))),
        ]
