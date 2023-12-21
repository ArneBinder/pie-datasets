import os
from typing import Dict

import pandas as pd
from pie_modules.document.processing import RegexPartitioner
from pytorch_ie.annotations import BinaryRelation
from pytorch_ie.documents import (
    TextDocumentWithLabeledSpansAndBinaryRelations,
    TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions,
)

from pie_datasets.builders import BratBuilder
from pie_datasets.builders.brat import BratConfig, BratDocumentWithMergedSpans
from pie_datasets.core.dataset import DocumentConvertersType
from pie_datasets.document.processing import Caster, Converter, Pipeline


def get_split_paths(url_split_ids: str, subdirectory: str) -> Dict[str, str]:
    df_splits = pd.read_csv(url_split_ids, sep=";")
    splits2ids = df_splits.groupby(df_splits["SET"]).agg(list).to_dict()["ID"]
    return {
        split.lower(): [os.path.join(subdirectory, split_id) for split_id in split_ids]
        for split, split_ids in splits2ids.items()
    }


URL = "https://github.com/ArneBinder/pie-datasets/raw/83fb46f904b13f335b6da3cce2fc7004d802ce4e/data/datasets/ArgumentAnnotatedEssays-2.0/brat-project-final.zip"
URL_SPLIT_IDS = "https://raw.githubusercontent.com/ArneBinder/pie-datasets/83fb46f904b13f335b6da3cce2fc7004d802ce4e/data/datasets/ArgumentAnnotatedEssays-2.0/train-test-split.csv"
SPLIT_PATHS = get_split_paths(URL_SPLIT_IDS, subdirectory="brat-project-final")

DEFAULT_ATTRIBUTIONS_TO_RELATIONS_DICT = {"For": "supports", "Against": "attacks"}


def convert_aae2_claim_attributions_to_relations(
    document: BratDocumentWithMergedSpans,
    method: str,
    attributions_to_relations_mapping: Dict[str, str] = DEFAULT_ATTRIBUTIONS_TO_RELATIONS_DICT,
    major_claim_label: str = "MajorClaim",
    claim_label: str = "Claim",
    semantically_same_label: str = "semantically_same",
) -> TextDocumentWithLabeledSpansAndBinaryRelations:
    """This function collects the attributions of Claims from BratDocumentWithMergedSpans, and
    build new relations between MajorClaims and Claims based on these attributions in the following
    way:
    1) "connect_first":
    Each Claim points to the first MajorClaim,
    and the other MajorClaim(s) is labeled as semantically same as the first MajorClaim.
    The number of new relations created are: NoOfMajorClaim - 1 + NoOfClaim.
    2) "connect_all":
    Each Claim points to every MajorClaim; creating many-to-many relations.
    The number of new relations created are: NoOfMajorClaim x NoOfClaim.

    The attributions are transformed into the relation labels as listed in
    DEFAULT_ATTRIBUTIONS_TO_RELATIONS_DICT dictionary.
    """
    document = document.copy()
    new_document = TextDocumentWithLabeledSpansAndBinaryRelations(
        text=document.text, id=document.id, metadata=document.metadata
    )
    # import from document
    spans = document.spans.clear()
    new_document.labeled_spans.extend(spans)
    relations = document.relations.clear()
    new_document.binary_relations.extend(relations)

    claim_attributes = [
        attribute
        for attribute in document.span_attributes
        if attribute.annotation.label == claim_label
    ]

    # get all MajorClaims
    # sorted by start position to ensure the first MajorClaim is really the first one that occurs in the text
    major_claims = sorted(
        [mc for mc in new_document.labeled_spans if mc.label == major_claim_label],
        key=lambda span: span.start,
    )

    if method == "connect_first":
        if len(major_claims) > 0:
            first_major_claim = major_claims.pop(0)

            # Add relation between Claims and first MajorClaim
            for claim_attribute in claim_attributes:
                new_relation = BinaryRelation(
                    head=claim_attribute.annotation,
                    tail=first_major_claim,
                    label=attributions_to_relations_mapping[claim_attribute.value],
                )
                new_document.binary_relations.append(new_relation)

            # Add relations between MajorClaims
            for majorclaim in major_claims:
                new_relation = BinaryRelation(
                    head=majorclaim,
                    tail=first_major_claim,
                    label=semantically_same_label,
                )
                new_document.binary_relations.append(new_relation)

    elif method == "connect_all":
        for major_claim in major_claims:
            for claim_attribute in claim_attributes:
                new_relation = BinaryRelation(
                    head=claim_attribute.annotation,
                    tail=major_claim,
                    label=attributions_to_relations_mapping[claim_attribute.value],
                )
                new_document.binary_relations.append(new_relation)

    else:
        raise ValueError(f"unknown method: {method}")

    return new_document


def get_common_pipeline_steps(conversion_method: str) -> dict:
    return dict(
        convert=Converter(
            function=convert_aae2_claim_attributions_to_relations,
            method=conversion_method,
        ),
    )


class ArgumentAnnotatedEssaysV2Config(BratConfig):
    def __init__(self, conversion_method: str, **kwargs):
        """BuilderConfig for ArgumentAnnotatedEssaysV2.

        Args:
            conversion_method: either "connect_first" or "connect_all", see convert_aae2_claim_attributions_to_relations
            **kwargs: keyword arguments forwarded to super.
        """
        super().__init__(**kwargs)
        self.conversion_method = conversion_method


class ArgumentAnnotatedEssaysV2(BratBuilder):
    BASE_DATASET_PATH = "DFKI-SLT/brat"
    BASE_DATASET_REVISION = "bb8c37d84ddf2da1e691d226c55fef48fd8149b5"

    # we need to add None to the list of dataset variants to support the default dataset variant
    BASE_BUILDER_KWARGS_DICT = {
        dataset_variant: {"url": URL, "split_paths": SPLIT_PATHS}
        for dataset_variant in ["default", "merge_fragmented_spans", None]
    }

    BUILDER_CONFIGS = [
        ArgumentAnnotatedEssaysV2Config(name="default", conversion_method="connect_first"),
        ArgumentAnnotatedEssaysV2Config(
            name="merge_fragmented_spans",
            merge_fragmented_spans=True,
            conversion_method="connect_first",
        ),
    ]

    @property
    def document_converters(self) -> DocumentConvertersType:
        if self.config.name == "default":
            # we do not support any auto-conversion for the default BratDocument for now
            return {}
        elif self.config.name == "merge_fragmented_spans":
            return {
                TextDocumentWithLabeledSpansAndBinaryRelations: Pipeline(
                    **get_common_pipeline_steps(conversion_method=self.config.conversion_method)
                ),
                TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions: Pipeline(
                    **get_common_pipeline_steps(conversion_method=self.config.conversion_method),
                    cast=Caster(
                        document_type=TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions
                    ),
                    add_partitions=RegexPartitioner(
                        partition_layer_name="labeled_partitions",
                        pattern="\n",
                        strip_whitespace=True,
                        verbose=False,
                    ),
                ),
            }
        else:
            raise ValueError(f"Unknown dataset variant: {self.config.name}")
