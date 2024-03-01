from pie_modules.document.processing import (
    RegexPartitioner,
    RelationArgumentSorter,
    SpansViaRelationMerger,
    TextSpanTrimmer,
)
from pie_modules.documents import (
    TextDocumentWithLabeledMultiSpansAndBinaryRelations,
    TextDocumentWithLabeledMultiSpansBinaryRelationsAndLabeledPartitions,
    TextDocumentWithLabeledSpansAndBinaryRelations,
    TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions,
)
from pytorch_ie.core import Document

from pie_datasets.builders import BratBuilder, BratConfig
from pie_datasets.builders.brat import BratDocument, BratDocumentWithMergedSpans
from pie_datasets.core.dataset import DocumentConvertersType
from pie_datasets.document.processing import Caster, Pipeline

URL = "http://data.dws.informatik.uni-mannheim.de/sci-arg/compiled_corpus.zip"
SPLIT_PATHS = {"train": "compiled_corpus"}


def get_common_converter_pipeline_steps(target_document_type: type[Document]) -> dict:
    return dict(
        cast=Caster(
            document_type=target_document_type,
            field_mapping={"spans": "labeled_spans", "relations": "binary_relations"},
        ),
        trim_adus=TextSpanTrimmer(layer="labeled_spans"),
        sort_symmetric_relation_arguments=RelationArgumentSorter(
            relation_layer="binary_relations",
            label_whitelist=["parts_of_same", "semantically_same"],
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
            document = SpansViaRelationMerger(
                relation_layer="relations",
                link_relation_label="parts_of_same",
                create_multi_spans=True,
                result_document_type=BratDocument,
                result_field_mapping={"spans": "spans", "relations": "relations"},
            )(document)
        return document

    @property
    def document_converters(self) -> DocumentConvertersType:
        regex_partitioner = RegexPartitioner(
            partition_layer_name="labeled_partitions",
            pattern="<([^>/]+)>.*</\\1>",
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
