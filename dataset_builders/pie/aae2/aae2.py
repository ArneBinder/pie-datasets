import os
from typing import Dict

import pandas as pd
from pie_documents.annotations import BinaryRelation
from pie_documents.document.processing import RegexPartitioner
from pie_documents.documents import (
    TextDocumentWithLabeledSpansAndBinaryRelations,
    TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions,
)

from pie_datasets.builders import BratBuilder
from pie_datasets.builders.brat import BratConfig, BratDocumentWithMergedSpans, BratSpan
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
        for attribute in document.attributes
        if isinstance(attribute.annotation, BratSpan) and attribute.annotation.label == claim_label
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


def remove_cross_partition_relations(
    document: TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions,
) -> TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions:
    # for each labeled_spans entry, get the labeled_partitions entry it belongs to
    labeled_span2partition = {}
    for labeled_span in document.labeled_spans:
        for partition in document.labeled_partitions:
            if partition.start <= labeled_span.start and labeled_span.end <= partition.end:
                labeled_span2partition[labeled_span] = partition
                break
        else:
            raise ValueError(f"Could not find partition for labeled_span: {labeled_span}")

    result = document.copy(with_annotations=True)
    idx = 0
    for relation in document.binary_relations:
        head_partition = labeled_span2partition[relation.head]
        tail_partition = labeled_span2partition[relation.tail]
        if head_partition != tail_partition:
            result.binary_relations.pop(idx)
        else:
            idx += 1
    return result


# def split_documents_into_partitions(
#    document: TextDocumentWithLabeledSpansAndBinaryRelations,
# ) -> TextDocumentWithLabeledSpansAndBinaryRelations:
#    raise NotImplementedError("split_documents_into_partitions is not implemented yet.")


def get_common_pipeline_steps_paragraphs(conversion_method: str) -> dict:
    return dict(
        **get_common_pipeline_steps(conversion_method=conversion_method),
        cast=Caster(document_type=TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions),
        add_partitions=RegexPartitioner(
            partition_layer_name="labeled_partitions",
            default_partition_label="paragraph",
            pattern="\n",
            strip_whitespace=True,
            verbose=False,
        ),
    )


class ArgumentAnnotatedEssaysV2Config(BratConfig):
    def __init__(self, conversion_method: str, **kwargs):
        """BuilderConfig for ArgumentAnnotatedEssaysV2.

        Args:
            conversion_method: either "connect_first" or "connect_all", see convert_aae2_claim_attributions_to_relations
            **kwargs: keyword arguments forwarded to super.
        """
        super().__init__(merge_fragmented_spans=True, **kwargs)
        self.conversion_method = conversion_method


class ArgumentAnnotatedEssaysV2(BratBuilder):
    BUILDER_CONFIG_CLASS = ArgumentAnnotatedEssaysV2Config
    BASE_DATASET_PATH = "DFKI-SLT/brat"
    BASE_DATASET_REVISION = "bb8c37d84ddf2da1e691d226c55fef48fd8149b5"

    BUILDER_CONFIGS = [
        ArgumentAnnotatedEssaysV2Config(
            name=BratBuilder.DEFAULT_CONFIG_NAME,
            conversion_method="connect_first",
        ),
        ArgumentAnnotatedEssaysV2Config(
            name="paragraphs",
            conversion_method="connect_all",
        ),
    ]

    # we need to add None to the list of dataset variants to support the default dataset variant
    BASE_BUILDER_KWARGS_DICT = {
        dataset_variant: {"url": URL, "split_paths": SPLIT_PATHS, "trust_remote_code": True}
        for dataset_variant in [None] + [config.name for config in BUILDER_CONFIGS]
    }

    DOCUMENT_TYPES = {config.name: BratDocumentWithMergedSpans for config in BUILDER_CONFIGS}

    @property
    def document_converters(self) -> DocumentConvertersType:
        if self.config.name in [None, "main_claim_connect_all", BratBuilder.DEFAULT_CONFIG_NAME]:
            return {
                TextDocumentWithLabeledSpansAndBinaryRelations: Pipeline(
                    **get_common_pipeline_steps(conversion_method=self.config.conversion_method)
                ),
                TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions: Pipeline(
                    **get_common_pipeline_steps_paragraphs(
                        conversion_method=self.config.conversion_method
                    )
                ),
            }
        elif self.config.name == "paragraphs":
            return {
                # return one document per paragraph
                # TextDocumentWithLabeledSpansAndBinaryRelations: Pipeline(
                #    **get_common_pipeline_steps_paragraphs(conversion_method=self.config.conversion_method),
                #    split_documents=Converter(function=split_documents_into_partitions),
                # ),
                # just remove the cross-paragraph relations
                TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions: Pipeline(
                    **get_common_pipeline_steps_paragraphs(
                        conversion_method=self.config.conversion_method
                    ),
                    remove_cross_partition_relations=Converter(
                        function=remove_cross_partition_relations
                    ),
                ),
            }

        else:
            raise ValueError(f"Unknown dataset variant: {self.config.name}")
