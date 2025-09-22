import dataclasses
import logging
from typing import Union

from pie_core import AnnotationLayer, Document, annotation_field
from pie_documents.document.processing import (
    RegexPartitioner,
    RelationArgumentSorter,
    SpansViaRelationMerger,
    TextSpanTrimmer,
)
from pie_documents.documents import (
    TextDocumentWithLabeledMultiSpansAndBinaryRelations,
    TextDocumentWithLabeledMultiSpansBinaryRelationsAndLabeledPartitions,
    TextDocumentWithLabeledSpansAndBinaryRelations,
    TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions,
)

from pie_datasets.builders import BratBuilder, BratConfig
from pie_datasets.builders.brat import (
    BratAttribute,
    BratDocument,
    BratDocumentWithMergedSpans,
    BratNote,
)
from pie_datasets.core.dataset import DocumentConvertersType
from pie_datasets.document.processing import Caster, Pipeline

logger = logging.getLogger(__name__)

URL = "http://data.dws.informatik.uni-mannheim.de/sci-arg/compiled_corpus.zip"
SPLIT_PATHS = {"train": "compiled_corpus"}


@dataclasses.dataclass
class ConvertedBratDocument(TextDocumentWithLabeledMultiSpansAndBinaryRelations):
    span_attributes: AnnotationLayer[BratAttribute] = annotation_field(
        target="labeled_multi_spans"
    )
    relation_attributes: AnnotationLayer[BratAttribute] = annotation_field(
        target="binary_relations"
    )
    notes: AnnotationLayer[BratNote] = annotation_field(
        targets=[
            "labeled_multi_spans",
            "binary_relations",
            "span_attributes",
            "relation_attributes",
        ]
    )


@dataclasses.dataclass
class ConvertedBratDocumentWithMergedSpans(TextDocumentWithLabeledSpansAndBinaryRelations):
    span_attributes: AnnotationLayer[BratAttribute] = annotation_field(target="labeled_spans")
    relation_attributes: AnnotationLayer[BratAttribute] = annotation_field(
        target="binary_relations"
    )
    notes: AnnotationLayer[BratNote] = annotation_field(
        targets=["labeled_spans", "binary_relations", "span_attributes", "relation_attributes"]
    )


def get_common_converter_pipeline_steps(target_document_type: type[Document]) -> dict:
    return dict(
        cast=Caster(
            document_type=target_document_type,
            field_mapping={"spans": "labeled_spans", "relations": "binary_relations"},
        ),
        trim_adus=TextSpanTrimmer(layer="labeled_spans"),
        sort_symmetric_relation_arguments=RelationArgumentSorter(
            relation_layer="binary_relations",
            label_whitelist=["parts_of_same", "semantically_same", "contradicts"],
        ),
    )


def get_common_converter_pipeline_steps_with_resolve_parts_of_same(
    target_document_type: type[Document],
) -> dict:
    return dict(
        cast=Caster(
            document_type=target_document_type,
            field_mapping={"spans": "labeled_multi_spans", "relations": "binary_relations"},
        ),
        trim_adus=TextSpanTrimmer(layer="labeled_multi_spans"),
        sort_symmetric_relation_arguments=RelationArgumentSorter(
            relation_layer="binary_relations",
            label_whitelist=["semantically_same"],
        ),
    )


def remove_duplicate_relations(document: Union[BratDocument, BratDocumentWithMergedSpans]) -> None:
    if len(document.relations) > len(set(document.relations)):
        added = set()
        i = 0
        while i < len(document.relations):
            relation = document.relations[i]
            if relation in added:
                logger.warning(f"doc_id={document.id}: Removing duplicate relation: {relation}")
                document.relations.pop(i)
            else:
                added.add(relation)
                i += 1


class SciArgConfig(BratConfig):
    def __init__(
        self,
        name: str,
        resolve_parts_of_same: bool = False,
        **kwargs,
    ):
        super().__init__(name=name, merge_fragmented_spans=True, **kwargs)
        self.resolve_parts_of_same = resolve_parts_of_same


class SciArg(BratBuilder):
    BASE_DATASET_PATH = "DFKI-SLT/brat"
    BASE_DATASET_REVISION = "844de61e8a00dc6a93fc29dc185f6e617131fbf1"

    # Overwrite the default config to merge the span fragments.
    # The span fragments in SciArg come just from the new line splits, so we can merge them.
    # Actual span fragments are annotated via "parts_of_same" relations.
    BUILDER_CONFIGS = [
        SciArgConfig(name=BratBuilder.DEFAULT_CONFIG_NAME),
        SciArgConfig(name="resolve_parts_of_same", resolve_parts_of_same=True),
    ]
    DOCUMENT_TYPES = {
        BratBuilder.DEFAULT_CONFIG_NAME: BratDocumentWithMergedSpans,
        "resolve_parts_of_same": BratDocument,
    }

    # we need to add None to the list of dataset variants to support the default dataset variant
    BASE_BUILDER_KWARGS_DICT = {
        dataset_variant: {"url": URL, "split_paths": SPLIT_PATHS}
        for dataset_variant in ["default", "resolve_parts_of_same", None]
    }

    def _generate_document(self, example, **kwargs):
        document = super()._generate_document(example, **kwargs)
        if self.config.resolve_parts_of_same:
            # we need to convert the document to a different type to be able to merge the spans:
            # SpansViaRelationMerger expects the spans to be of type LabeledSpan,
            # but the document has spans of type BratSpan
            converted_doc = document.as_type(
                ConvertedBratDocumentWithMergedSpans,
                field_mapping={
                    "spans": "labeled_spans",
                    "relations": "binary_relations",
                },
                keep_remaining=True,
            )
            merged_document = SpansViaRelationMerger(
                relation_layer="binary_relations",
                link_relation_label="parts_of_same",
                create_multi_spans=True,
                result_document_type=ConvertedBratDocument,
                result_field_mapping={
                    "labeled_spans": "labeled_multi_spans",
                    "binary_relations": "binary_relations",
                    "span_attributes": "span_attributes",
                    "relation_attributes": "relation_attributes",
                    "notes": "notes",
                },
            )(converted_doc)
            # convert back to BratDocument
            document = merged_document.as_type(
                BratDocument,
                field_mapping={"labeled_multi_spans": "spans", "binary_relations": "relations"},
                keep_remaining=True,
            )
        else:
            # some documents have duplicate relations, remove them
            remove_duplicate_relations(document)

        return document

    @property
    def document_converters(self) -> DocumentConvertersType:
        regex_partitioner = RegexPartitioner(
            partition_layer_name="labeled_partitions",
            # find matching tags, allow newlines in between (s flag) and capture the tag name
            pattern="<([^>/]+)>(?s:.)*?</\\1>",
            label_group_id=1,
            label_whitelist=["Title", "Abstract", "H1"],
            skip_initial_partition=True,
            strip_whitespace=True,
        )
        if not self.config.resolve_parts_of_same:
            return {
                TextDocumentWithLabeledSpansAndBinaryRelations: Pipeline(
                    **get_common_converter_pipeline_steps(
                        TextDocumentWithLabeledSpansAndBinaryRelations
                    )
                ),
                TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions: Pipeline(
                    **get_common_converter_pipeline_steps(
                        TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions
                    ),
                    add_partitions=regex_partitioner,
                ),
            }
        else:
            return {
                #        TextDocumentWithLabeledSpansAndBinaryRelations: Pipeline(
                #            **get_common_converter_pipeline_steps_with_resolve_parts_of_same(
                #                TextDocumentWithLabeledSpansAndBinaryRelations
                #            )
                #        ),
                #        TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions: Pipeline(
                #            **get_common_converter_pipeline_steps_with_resolve_parts_of_same(
                #                TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions
                #            ),
                #            add_partitions=regex_partitioner,
                #        ),
                TextDocumentWithLabeledMultiSpansAndBinaryRelations: Pipeline(
                    **get_common_converter_pipeline_steps_with_resolve_parts_of_same(
                        TextDocumentWithLabeledMultiSpansAndBinaryRelations
                    )
                ),
                TextDocumentWithLabeledMultiSpansBinaryRelationsAndLabeledPartitions: Pipeline(
                    **get_common_converter_pipeline_steps_with_resolve_parts_of_same(
                        TextDocumentWithLabeledMultiSpansBinaryRelationsAndLabeledPartitions
                    ),
                    add_partitions=regex_partitioner,
                ),
            }
