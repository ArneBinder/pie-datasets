from pie_modules.document.processing import (
    RegexPartitioner,
    RelationArgumentSorter,
    TextSpanTrimmer,
)
from pytorch_ie.core import Document
from pytorch_ie.documents import (
    TextDocumentWithLabeledSpansAndBinaryRelations,
    TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions,
)

from pie_datasets.builders import BratBuilder, BratConfig
from pie_datasets.builders.brat import BratDocumentWithMergedSpans
from pie_datasets.document.processing import Caster, Pipeline

URL = "http://data.dws.informatik.uni-mannheim.de/sci-arg/compiled_corpus.zip"
SPLIT_PATHS = {"train": "compiled_corpus"}


def get_common_pipeline_steps(target_document_type: type[Document]) -> dict:
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


class SciArg(BratBuilder):
    BASE_DATASET_PATH = "DFKI-SLT/brat"
    BASE_DATASET_REVISION = "844de61e8a00dc6a93fc29dc185f6e617131fbf1"

    # Overwrite the default config to merge the span fragments.
    # The span fragments in SciArg come just from the new line splits, so we can merge them.
    # Actual span fragments are annotated via "parts_of_same" relations.
    BUILDER_CONFIGS = [
        BratConfig(name=BratBuilder.DEFAULT_CONFIG_NAME, merge_fragmented_spans=True),
    ]
    DOCUMENT_TYPES = {
        BratBuilder.DEFAULT_CONFIG_NAME: BratDocumentWithMergedSpans,
    }

    # we need to add None to the list of dataset variants to support the default dataset variant
    BASE_BUILDER_KWARGS_DICT = {
        dataset_variant: {"url": URL, "split_paths": SPLIT_PATHS}
        for dataset_variant in ["default", "merge_fragmented_spans", None]
    }

    DOCUMENT_CONVERTERS = {
        TextDocumentWithLabeledSpansAndBinaryRelations: Pipeline(
            **get_common_pipeline_steps(TextDocumentWithLabeledSpansAndBinaryRelations)
        ),
        TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions: Pipeline(
            **get_common_pipeline_steps(
                TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions
            ),
            add_partitions=RegexPartitioner(
                partition_layer_name="labeled_partitions",
                pattern="<([^>/]+)>.*</\\1>",
                label_group_id=1,
                label_whitelist=["Title", "Abstract", "H1"],
                skip_initial_partition=True,
                strip_whitespace=True,
            ),
        ),
    }
