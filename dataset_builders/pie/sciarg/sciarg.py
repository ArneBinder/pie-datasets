from pytorch_ie.core import Document
from pytorch_ie.documents import (
    TextDocumentWithLabeledSpansAndBinaryRelations,
    TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions,
)

from pie_datasets.builders import BratBuilder
from pie_datasets.core.dataset import DocumentConvertersType
from pie_datasets.document.processing import (
    Caster,
    Pipeline,
    RegexPartitioner,
    RelationArgumentSorter,
    TextSpanTrimmer,
)

URL = "http://data.dws.informatik.uni-mannheim.de/sci-arg/compiled_corpus.zip"
SUBDIRECTORY_MAPPING = {"compiled_corpus": "train"}


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
    # we need to add None to the list of dataset variants to support the default dataset variant
    BASE_BUILDER_KWARGS_DICT = {
        dataset_variant: {"url": URL, "subdirectory_mapping": SUBDIRECTORY_MAPPING}
        for dataset_variant in ["default", "merge_fragmented_spans", None]
    }

    @property
    def document_converters(self) -> DocumentConvertersType:
        if self.config.name == "default":
            return {}
        elif self.config.name == "merge_fragmented_spans":
            return {
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
        else:
            raise ValueError(f"Unknown dataset variant: {self.config.name}")
