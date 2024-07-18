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
        assert hf_example["pmid"] == "10471277"
        assert hf_example["text"].startswith(
            "Probing the salmeterol binding site on the beta 2-adrenergic receptor"
        )

        # check entities: first and last example
        assert len(hf_example["entities"]["id"]) == 43
        assert [
            hf_example["entities"]["type"][0],
            hf_example["entities"]["id"][0],
            hf_example["entities"]["text"][0],
            hf_example["entities"]["offsets"][0],
        ] == ["CHEMICAL", "T1", "Salmeterol", [135, 145]]
        assert [
            hf_example["entities"]["type"][42],
            hf_example["entities"]["id"][42],
            hf_example["entities"]["text"][42],
            hf_example["entities"]["offsets"][42],
        ] == ["GENE-Y", "T44", "adenylyl cyclase", [761, 777]]

        # check relations: first and last example
        assert len(hf_example["relations"]["type"]) == 20
        assert [
            hf_example["relations"]["type"][0],
            hf_example["relations"]["arg1"][0],
            hf_example["relations"]["arg2"][0],
        ] == ["CPR:2", "T30", "T33"]
        assert [
            hf_example["relations"]["type"][19],
            hf_example["relations"]["arg1"][19],
            hf_example["relations"]["arg2"][19],
        ] == ["CPR:2", "T3", "T34"]

    elif dataset_variant == "chemprot_bigbio_kb":
        assert hf_example["id"] == "0"
        assert hf_example["document_id"] == "10471277"
        assert hf_example["events"] == []
        assert hf_example["coreferences"] == []
        # check passages
        assert len(hf_example["passages"]) == 1
        assert hf_example["passages"][0] == {
            "id": "1",
            "type": "title and abstract",
            "text": [
                'Probing the salmeterol binding site on the beta 2-adrenergic receptor using a novel photoaffinity ligand, [(125)I]iodoazidosalmeterol.\nSalmeterol is a long-acting beta2-adrenergic receptor (beta 2AR) agonist used clinically to treat asthma. In addition to binding at the active agonist site, it has been proposed that salmeterol also binds with very high affinity at a second site, termed the "exosite", and that this exosite contributes to the long duration of action of salmeterol. To determine the position of the phenyl ring of the aralkyloxyalkyl side chain of salmeterol in the beta 2AR binding site, we designed and synthesized the agonist photoaffinity label [(125)I]iodoazidosalmeterol ([125I]IAS). In direct adenylyl cyclase activation, in effects on adenylyl cyclase after pretreatment of intact cells, and in guinea pig tracheal relaxation assays, IAS and the parent drug salmeterol behave essentially the same. Significantly, the photoreactive azide of IAS is positioned on the phenyl ring at the end of the molecule which is thought to be involved in exosite binding. Carrier-free radioiodinated [125I]IAS was used to photolabel epitope-tagged human beta 2AR in membranes prepared from stably transfected HEK 293 cells. Labeling with [(125)I]IAS was blocked by 10 microM (-)-alprenolol and inhibited by addition of GTP gamma S, and [125I]IAS migrated at the same position on an SDS-PAGE gel as the beta 2AR labeled by the antagonist photoaffinity label [125I]iodoazidobenzylpindolol ([125I]IABP). The labeled receptor was purified on a nickel affinity column and cleaved with factor Xa protease at a specific sequence in the large loop between transmembrane segments 5 and 6, yielding two peptides. While the control antagonist photoaffinity label [125I]IABP labeled both the large N-terminal fragment [containing transmembranes (TMs) 1-5] and the smaller C-terminal fragment (containing TMs 6 and 7), essentially all of the [125I]IAS labeling was on the smaller C-terminal peptide containing TMs 6 and 7. This direct biochemical evidence demonstrates that when salmeterol binds to the receptor, its hydrophobic aryloxyalkyl tail is positioned near TM 6 and/or TM 7. A model of IAS binding to the beta 2AR is proposed.'
            ],
            "offsets": [[0, 2232]],
        }

        # check entities: first and last example
        assert len(hf_example["entities"]) == 43
        assert hf_example["entities"][0] == {
            "id": "2",
            "type": "CHEMICAL",
            "text": ["Salmeterol"],
            "offsets": [[135, 145]],
            "normalized": [],
        }
        assert hf_example["entities"][42] == {
            "id": "44",
            "type": "GENE-Y",
            "text": ["adenylyl cyclase"],
            "offsets": [[761, 777]],
            "normalized": [],
        }

        # check relations: first and last example
        assert len(hf_example["relations"]) == 20
        assert hf_example["relations"][0] == {
            "id": "45",
            "type": "Regulator",
            "arg1_id": "31",
            "arg2_id": "33",
            "normalized": [],
        }
        assert hf_example["relations"][19] == {
            "id": "64",
            "type": "Regulator",
            "arg1_id": "4",
            "arg2_id": "34",
            "normalized": [],
        }

    elif dataset_variant == "chemprot_shared_task_eval_source":
        assert hf_example["pmid"] == "10471277"
        assert hf_example["text"].startswith(
            "Probing the salmeterol binding site on the beta 2-adrenergic receptor"
        )

        # check entities: first and last example
        assert len(hf_example["entities"]["id"]) == 43
        assert [
            hf_example["entities"]["type"][0],
            hf_example["entities"]["id"][0],
            hf_example["entities"]["text"][0],
            hf_example["entities"]["offsets"][0],
        ] == ["CHEMICAL", "T1", "Salmeterol", [135, 145]]
        assert [
            hf_example["entities"]["type"][42],
            hf_example["entities"]["id"][42],
            hf_example["entities"]["text"][42],
            hf_example["entities"]["offsets"][42],
        ] == ["GENE-Y", "T44", "adenylyl cyclase", [761, 777]]

        # check relations: first and last example
        assert len(hf_example["relations"]["type"]) == 10
        assert [
            hf_example["relations"]["type"][0],
            hf_example["relations"]["arg1"][0],
            hf_example["relations"]["arg2"][0],
        ] == ["CPR:5", "T1", "T37"]
        assert [
            hf_example["relations"]["type"][9],
            hf_example["relations"]["arg1"][9],
            hf_example["relations"]["arg2"][9],
        ] == ["CPR:6", "T7", "T34"]

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
