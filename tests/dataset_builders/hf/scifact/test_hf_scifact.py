import json
import os
from collections import defaultdict
from typing import Any, Dict, List

import pytest
from datasets import DownloadManager, load_dataset

from dataset_builders.hf.scifact.scifact import DATA_URL, SUBDIR, SciFact, VARIANT_DOCUMENTS, VARIANT_CLAIMS
from tests import FIXTURES_ROOT
from tests.dataset_builders.common import HF_BASE_PATH

DATASET_NAME = "scifact"
HF_DATASET_PATH = HF_BASE_PATH / DATASET_NAME
FIXTURES_DATASET_PATH = FIXTURES_ROOT / "dataset_builders" / "hf" / DATASET_NAME
SPLIT_NAMES = {"train", "validation", "test"}
# same size for all splits since we use all documents (complete corpus) in each settings, only the claims/evidence
# differ
SPLIT_SIZES = {
    VARIANT_DOCUMENTS: {
        "train": 5184,
        "validation": 5184,
        "test": 5184,
    },
    VARIANT_CLAIMS: {"train": 809, "test": 300, "validation": 300},
}
# we consider this sample because it contains evidence for different claims
EXAMPLE_DOC_ID = 6


@pytest.fixture(params=["train", "validation", "test"], scope="module")
def split_name(request):
    return request.param


@pytest.fixture(params=[config.name for config in SciFact.BUILDER_CONFIGS], scope="module")
def config_name(request):
    return request.param


@pytest.fixture(scope="module")
def hf_dataset(config_name):
    return load_dataset(str(HF_DATASET_PATH), name=config_name)


def dict_of_lists_to_list_of_dicts(dict_of_lists):
    keys = dict_of_lists.keys()
    values = zip(*dict_of_lists.values())
    return [dict(zip(keys, v)) for v in values]


def test_hf_dataset(hf_dataset, config_name):
    assert set(hf_dataset) == SPLIT_NAMES
    split_sizes = {split_name: len(ds) for split_name, ds in hf_dataset.items()}
    assert split_sizes == SPLIT_SIZES[config_name]
    all_claims = defaultdict(list)
    for split_name, split in hf_dataset.items():
        for ex in split:
            ex.copy()  # check that examples are serializable
            if config_name == VARIANT_DOCUMENTS:
                all_claims[split_name] += ex["claims"]["claim"]
            elif config_name == VARIANT_CLAIMS:
                all_claims[split_name].append(ex["claim"])
            else:
                raise ValueError(f"Unknown dataset variant: {config_name}")
    num_claims_unique = {split_name: len(set(claims)) for split_name, claims in all_claims.items()}
    assert num_claims_unique == {"test": 297, "train": 807, "validation": 300}


@pytest.fixture(scope="module")
def hf_example(hf_dataset, split_name):
    return hf_dataset[split_name][EXAMPLE_DOC_ID]


def test_hf_example(hf_example, config_name, split_name):
    if config_name == VARIANT_DOCUMENTS:
        if split_name == "train":
            assert hf_example == {
                "doc_id": 33370,
                "title": "Targeting A20 Decreases Glioma Stem Cell Survival and Tumor Growth",
                "abstract": [
                    "Glioblastomas are deadly cancers that display a functional cellular hierarchy maintained by self-renewing glioblastoma stem cells (GSCs).",
                    "GSCs are regulated by molecular pathways distinct from the bulk tumor that may be useful therapeutic targets.",
                    "We determined that A20 (TNFAIP3), a regulator of cell survival and the NF-kappaB pathway, is overexpressed in GSCs relative to non-stem glioblastoma cells at both the mRNA and protein levels.",
                    "To determine the functional significance of A20 in GSCs, we targeted A20 expression with lentiviral-mediated delivery of short hairpin RNA (shRNA).",
                    "Inhibiting A20 expression decreased GSC growth and survival through mechanisms associated with decreased cell-cycle progression and decreased phosphorylation of p65/RelA. Elevated levels of A20 in GSCs contributed to apoptotic resistance: GSCs were less susceptible to TNFalpha-induced cell death than matched non-stem glioma cells, but A20 knockdown sensitized GSCs to TNFalpha-mediated apoptosis.",
                    "The decreased survival of GSCs upon A20 knockdown contributed to the reduced ability of these cells to self-renew in primary and secondary neurosphere formation assays.",
                    "The tumorigenic potential of GSCs was decreased with A20 targeting, resulting in increased survival of mice bearing human glioma xenografts.",
                    "In silico analysis of a glioma patient genomic database indicates that A20 overexpression and amplification is inversely correlated with survival.",
                    "Together these data indicate that A20 contributes to glioma maintenance through effects on the glioma stem cell subpopulation.",
                    "Although inactivating mutations in A20 in lymphoma suggest A20 can act as a tumor suppressor, similar point mutations have not been identified through glioma genomic sequencing: in fact, our data suggest A20 may function as a tumor enhancer in glioma through promotion of GSC survival.",
                    "A20 anticancer therapies should therefore be viewed with caution as effects will likely differ depending on the tumor type.",
                ],
                "structured": False,
                "claims": {
                    "id": [1134, 1135, 1136, 1403, 1404],
                    "claim": [
                        "TNFAIP3 is a glioblastoma tumor enhancer.",
                        "TNFAIP3 is a glioblastoma tumor suppressor.",
                        "TNFAIP3 is a tumor enhancer in glioblastoma.",
                        "siRNA knockdown of A20 accelerates tumor progression in an in vivo murine xenograft model.",
                        "siRNA knockdown of A20 slows tumor progression in an in vivo murine xenograft model.",
                    ],
                    "evidence": [
                        {"label": ["SUPPORT", "SUPPORT"], "sentences": [[4], [9]]},
                        {"label": ["CONTRADICT", "CONTRADICT"], "sentences": [[4], [9]]},
                        {"label": ["SUPPORT", "SUPPORT"], "sentences": [[6], [9]]},
                        {"label": ["CONTRADICT"], "sentences": [[6]]},
                        {"label": ["SUPPORT"], "sentences": [[6]]},
                    ],
                },
            }
        elif split_name == "validation":
            assert hf_example == {
                "doc_id": 33370,
                "title": "Targeting A20 Decreases Glioma Stem Cell Survival and Tumor Growth",
                "abstract": [
                    "Glioblastomas are deadly cancers that display a functional cellular hierarchy maintained by self-renewing glioblastoma stem cells (GSCs).",
                    "GSCs are regulated by molecular pathways distinct from the bulk tumor that may be useful therapeutic targets.",
                    "We determined that A20 (TNFAIP3), a regulator of cell survival and the NF-kappaB pathway, is overexpressed in GSCs relative to non-stem glioblastoma cells at both the mRNA and protein levels.",
                    "To determine the functional significance of A20 in GSCs, we targeted A20 expression with lentiviral-mediated delivery of short hairpin RNA (shRNA).",
                    "Inhibiting A20 expression decreased GSC growth and survival through mechanisms associated with decreased cell-cycle progression and decreased phosphorylation of p65/RelA. Elevated levels of A20 in GSCs contributed to apoptotic resistance: GSCs were less susceptible to TNFalpha-induced cell death than matched non-stem glioma cells, but A20 knockdown sensitized GSCs to TNFalpha-mediated apoptosis.",
                    "The decreased survival of GSCs upon A20 knockdown contributed to the reduced ability of these cells to self-renew in primary and secondary neurosphere formation assays.",
                    "The tumorigenic potential of GSCs was decreased with A20 targeting, resulting in increased survival of mice bearing human glioma xenografts.",
                    "In silico analysis of a glioma patient genomic database indicates that A20 overexpression and amplification is inversely correlated with survival.",
                    "Together these data indicate that A20 contributes to glioma maintenance through effects on the glioma stem cell subpopulation.",
                    "Although inactivating mutations in A20 in lymphoma suggest A20 can act as a tumor suppressor, similar point mutations have not been identified through glioma genomic sequencing: in fact, our data suggest A20 may function as a tumor enhancer in glioma through promotion of GSC survival.",
                    "A20 anticancer therapies should therefore be viewed with caution as effects will likely differ depending on the tumor type.",
                ],
                "structured": False,
                "claims": {
                    "id": [1137],
                    "claim": ["TNFAIP3 is a tumor suppressor in glioblastoma."],
                    "evidence": [{"label": ["CONTRADICT", "CONTRADICT"], "sentences": [[6], [9]]}],
                },
            }
        elif split_name == "test":
            assert hf_example == {
                "doc_id": 33370,
                "title": "Targeting A20 Decreases Glioma Stem Cell Survival and Tumor Growth",
                "abstract": [
                    "Glioblastomas are deadly cancers that display a functional cellular hierarchy maintained by self-renewing glioblastoma stem cells (GSCs).",
                    "GSCs are regulated by molecular pathways distinct from the bulk tumor that may be useful therapeutic targets.",
                    "We determined that A20 (TNFAIP3), a regulator of cell survival and the NF-kappaB pathway, is overexpressed in GSCs relative to non-stem glioblastoma cells at both the mRNA and protein levels.",
                    "To determine the functional significance of A20 in GSCs, we targeted A20 expression with lentiviral-mediated delivery of short hairpin RNA (shRNA).",
                    "Inhibiting A20 expression decreased GSC growth and survival through mechanisms associated with decreased cell-cycle progression and decreased phosphorylation of p65/RelA. Elevated levels of A20 in GSCs contributed to apoptotic resistance: GSCs were less susceptible to TNFalpha-induced cell death than matched non-stem glioma cells, but A20 knockdown sensitized GSCs to TNFalpha-mediated apoptosis.",
                    "The decreased survival of GSCs upon A20 knockdown contributed to the reduced ability of these cells to self-renew in primary and secondary neurosphere formation assays.",
                    "The tumorigenic potential of GSCs was decreased with A20 targeting, resulting in increased survival of mice bearing human glioma xenografts.",
                    "In silico analysis of a glioma patient genomic database indicates that A20 overexpression and amplification is inversely correlated with survival.",
                    "Together these data indicate that A20 contributes to glioma maintenance through effects on the glioma stem cell subpopulation.",
                    "Although inactivating mutations in A20 in lymphoma suggest A20 can act as a tumor suppressor, similar point mutations have not been identified through glioma genomic sequencing: in fact, our data suggest A20 may function as a tumor enhancer in glioma through promotion of GSC survival.",
                    "A20 anticancer therapies should therefore be viewed with caution as effects will likely differ depending on the tumor type.",
                ],
                "structured": False,
                "claims": {"id": [], "claim": [], "evidence": []},
            }
        else:
            raise ValueError(f"Unknown split name: {split_name}")
    elif config_name == VARIANT_CLAIMS:
        if split_name == "train":
            assert hf_example == {
                "id": 11,
                "claim": "4-PBA treatment raises endoplasmic reticulum stress in response to general endoplasmic reticulum stress markers.",
                "cited_docs": {
                    "doc_id": [32587939],
                    "title": [
                        "Wolfram syndrome 1 and adenylyl cyclase 8 interact at the plasma membrane to regulate insulin production and secretion"
                    ],
                    "abstract": [
                        [
                            "Endoplasmic reticulum (ER) stress causes pancreatic β-cell dysfunction and contributes to β-cell loss and the progression of type 2 diabetes.",
                            "Wolfram syndrome 1 (WFS1) has been shown to be an important regulator of the ER stress signalling pathway; however, its role in β-cell function remains unclear.",
                            "Here we provide evidence that WFS1 is essential for glucose- and glucagon-like peptide 1 (GLP-1)-stimulated cyclic AMP production and regulation of insulin biosynthesis and secretion.",
                            "Stimulation with glucose causes WFS1 translocation from the ER to the plasma membrane, where it forms a complex with adenylyl cyclase 8 (AC8), an essential cAMP-generating enzyme in the β-cell that integrates glucose and GLP-1 signalling.",
                            "ER stress and mutant WFS1 inhibit complex formation and activation of AC8, reducing cAMP synthesis and insulin secretion.",
                            "These findings reveal that an ER-stress-related protein has a distinct role outside the ER regulating both insulin biosynthesis and secretion.",
                            "The reduction of WFS1 protein on the plasma membrane during ER stress is a contributing factor for β-cell dysfunction and progression of type 2 diabetes.",
                        ]
                    ],
                    "structured": [False],
                    "evidence": [{"label": [], "sentences": []}],
                },
            }
        elif split_name == "validation":
            assert hf_example == {
                "id": 48,
                "claim": "A total of 1,000 people in the UK are asymptomatic carriers of vCJD infection.",
                "cited_docs": {
                    "doc_id": [13734012],
                    "title": [
                        "Prevalent abnormal prion protein in human appendixes after bovine spongiform encephalopathy epizootic: large scale survey"
                    ],
                    "abstract": [
                        [
                            "OBJECTIVES To carry out a further survey of archived appendix samples to understand better the differences between existing estimates of the prevalence of subclinical infection with prions after the bovine spongiform encephalopathy epizootic and to see whether a broader birth cohort was affected, and to understand better the implications for the management of blood and blood products and for the handling of surgical instruments.   \n",
                            "DESIGN Irreversibly unlinked and anonymised large scale survey of archived appendix samples.   \n",
                            "SETTING Archived appendix samples from the pathology departments of 41 UK hospitals participating in the earlier survey, and additional hospitals in regions with lower levels of participation in that survey.   ",
                            "SAMPLE 32,441 archived appendix samples fixed in formalin and embedded in paraffin and tested for the presence of abnormal prion protein (PrP).   \n",
                            "RESULTS Of the 32,441 appendix samples 16 were positive for abnormal PrP, indicating an overall prevalence of 493 per million population (95% confidence interval 282 to 801 per million).",
                            "The prevalence in those born in 1941-60 (733 per million, 269 to 1596 per million) did not differ significantly from those born between 1961 and 1985 (412 per million, 198 to 758 per million) and was similar in both sexes and across the three broad geographical areas sampled.",
                            "Genetic testing of the positive specimens for the genotype at PRNP codon 129 revealed a high proportion that were valine homozygous compared with the frequency in the normal population, and in stark contrast with confirmed clinical cases of vCJD, all of which were methionine homozygous at PRNP codon 129.   \n",
                            "CONCLUSIONS This study corroborates previous studies and suggests a high prevalence of infection with abnormal PrP, indicating vCJD carrier status in the population compared with the 177 vCJD cases to date.",
                            "These findings have important implications for the management of blood and blood products and for the handling of surgical instruments.",
                        ]
                    ],
                    "structured": [True],
                    "evidence": [{"label": [], "sentences": []}],
                },
            }
        elif split_name == "test":
            assert hf_example == {
                "cited_docs": {
                    "abstract": [],
                    "doc_id": [],
                    "evidence": [],
                    "structured": [],
                    "title": [],
                },
                "claim": "A deficiency of folate decreases blood levels of homocysteine.",
                "id": 33,
            }
        else:
            raise ValueError(f"Unknown split name: {split_name}")
    else:
        raise ValueError(f"Unknown dataset variant: {config_name}")


def compare_original_and_converted_to_output_eval_data(
    original_claim_data: List[Dict[str, Any]], converted_output_data: List[Dict[str, Any]]
):
    assert len(converted_output_data) == len(original_claim_data)
    for idx, original_claim_annotation in enumerate(original_claim_data):
        converted_claim_annotation = converted_output_data[idx]
        assert original_claim_annotation["id"] == converted_claim_annotation["id"]
        for doc_id in original_claim_annotation["evidence"]:
            assert int(doc_id) in converted_claim_annotation["evidence"]
            # Take the first element because the original data are annotated
            # with evidence sets that we do not use in the output format
            assert (
                original_claim_annotation["evidence"][doc_id][0]["label"]
                == converted_claim_annotation["evidence"][int(doc_id)]["label"]
            )
            original_evidence_sentences = []
            for ev in original_claim_annotation["evidence"][doc_id]:
                original_evidence_sentences.extend(ev["sentences"])
            sorted_original_sentences = sorted(original_evidence_sentences)
            sorted_converted_sentences = sorted(
                converted_claim_annotation["evidence"][int(doc_id)]["sentences"]
            )
            assert sorted_original_sentences == sorted_converted_sentences


def test_convert_to_output_eval_format(config_name):
    # Check that conversion of the HF dataset to the required output format works
    input_data = load_dataset(
        path=str(HF_DATASET_PATH),
        name=config_name,
        data_dir=str(FIXTURES_DATASET_PATH),
        split="train",
    )
    builder = SciFact(name=config_name)

    if config_name == VARIANT_DOCUMENTS:
        converted_output_data = builder._convert_to_output_eval_format(input_data)

        claims_filepath = str(FIXTURES_DATASET_PATH / "claims_train.jsonl")
        with open(claims_filepath) as f:
            original_claim_data = [json.loads(line) for line in f.readlines()]

        compare_original_and_converted_to_output_eval_data(original_claim_data, converted_output_data)
    elif config_name == VARIANT_CLAIMS:
        with pytest.raises(NotImplementedError) as exc_info:
            builder._convert_to_output_eval_format(input_data)
        assert str(exc_info.value) == f"_convert_to_output_eval_format is not yet implemented for dataset variant {config_name}"
    else:
        raise ValueError(f"Unknown dataset variant: {config_name}")


@pytest.mark.slow
def test_convert_to_output_eval_format_all(config_name):
    # Check that conversion of the HF dataset to the required output format works
    input_data = load_dataset(str(HF_DATASET_PATH), name=config_name, split="train")
    builder = SciFact(name=config_name)

    if config_name == VARIANT_DOCUMENTS:
        converted_output_data = builder._convert_to_output_eval_format(input_data)

        dl_manager = DownloadManager()
        data_dir = os.path.join(dl_manager.download_and_extract(DATA_URL), SUBDIR)
        claims_filepath = os.path.join(data_dir, "claims_train.jsonl")
        with open(claims_filepath) as f:
            original_claim_data = [json.loads(line) for line in f.readlines()]

        compare_original_and_converted_to_output_eval_data(original_claim_data, converted_output_data)
    elif config_name == VARIANT_CLAIMS:
        with pytest.raises(NotImplementedError) as exc_info:
            builder._convert_to_output_eval_format(input_data)
        assert str(exc_info.value) == f"_convert_to_output_eval_format is not yet implemented for dataset variant {config_name}"
    else:
        raise ValueError(f"Unknown dataset variant: {config_name}")
