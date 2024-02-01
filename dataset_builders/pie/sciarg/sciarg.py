import logging
from typing import Sequence, Set, Tuple, Union

import networkx as nx
from pie_modules.document.processing import (
    RegexPartitioner,
    RelationArgumentSorter,
    TextSpanTrimmer,
)
from pytorch_ie.annotations import BinaryRelation, LabeledMultiSpan, LabeledSpan
from pytorch_ie.core import Document
from pytorch_ie.documents import (
    TextDocumentWithLabeledSpansAndBinaryRelations,
    TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions,
)

from pie_datasets.builders import BratBuilder, BratConfig
from pie_datasets.builders.brat import BratDocumentWithMergedSpans
from pie_datasets.core.dataset import DocumentConvertersType
from pie_datasets.document.processing import Caster, Pipeline

URL = "http://data.dws.informatik.uni-mannheim.de/sci-arg/compiled_corpus.zip"
SPLIT_PATHS = {"train": "compiled_corpus"}


logger = logging.getLogger(__name__)


def _merge_spans_via_relation(
    spans: Sequence[LabeledSpan],
    relations: Sequence[BinaryRelation],
    link_relation_label: str,
    create_multi_spans: bool = True,
) -> Tuple[Union[Set[LabeledSpan], Set[LabeledMultiSpan]], Set[BinaryRelation]]:
    # convert list of relations to a graph to easily calculate connected components to merge
    g = nx.Graph()
    link_relations = []
    other_relations = []
    for rel in relations:
        if rel.label == link_relation_label:
            link_relations.append(rel)
            # never merge spans that have not the same label
            if (
                not (isinstance(rel.head, LabeledSpan) or isinstance(rel.tail, LabeledSpan))
                or rel.head.label == rel.tail.label
            ):
                g.add_edge(rel.head, rel.tail)
            else:
                logger.debug(
                    f"spans to merge do not have the same label, do not merge them: {rel.head}, {rel.tail}"
                )
        else:
            other_relations.append(rel)

    span_mapping = {}
    connected_components: Set[LabeledSpan]
    for connected_components in nx.connected_components(g):
        # all spans in a connected component have the same label
        label = list(span.label for span in connected_components)[0]
        connected_components_sorted = sorted(connected_components, key=lambda span: span.start)
        if create_multi_spans:
            new_span = LabeledMultiSpan(
                slices=tuple((span.start, span.end) for span in connected_components_sorted),
                label=label,
            )
        else:
            new_span = LabeledSpan(
                start=min(span.start for span in connected_components_sorted),
                end=max(span.end for span in connected_components_sorted),
                label=label,
            )
        for span in connected_components_sorted:
            span_mapping[span] = new_span
    for span in spans:
        if span not in span_mapping:
            if create_multi_spans:
                span_mapping[span] = LabeledMultiSpan(
                    slices=((span.start, span.end),), label=span.label, score=span.score
                )
            else:
                span_mapping[span] = LabeledSpan(
                    start=span.start, end=span.end, label=span.label, score=span.score
                )

    new_spans = set(span_mapping.values())
    new_relations = {
        BinaryRelation(
            head=span_mapping[rel.head],
            tail=span_mapping[rel.tail],
            label=rel.label,
            score=rel.score,
        )
        for rel in other_relations
    }

    return new_spans, new_relations


class SpansWithRelationsMerger:
    """Merge spans that are connected via a specific relation type.

    Args:
        relation_layer: The name of the layer that contains the relations.
        link_relation_label: The label of the relations that connect the spans.
        create_multi_spans: If True, the merged spans are LabeledMultiSpans, otherwise LabeledSpans.
    """

    def __init__(
        self,
        relation_layer: str,
        link_relation_label: str,
        result_document_type: type[Document],
        result_field_mapping: dict[str, str],
        create_multi_spans: bool = True,
    ):
        self.relation_layer = relation_layer
        self.link_relation_label = link_relation_label
        self.create_multi_spans = create_multi_spans
        self.result_document_type = result_document_type
        self.result_field_mapping = result_field_mapping

    def __call__(self, document: Document) -> Document:
        relations: Sequence[BinaryRelation] = document[self.relation_layer]
        spans: Sequence[LabeledSpan] = document[self.relation_layer].target_layer

        new_spans, new_relations = _merge_spans_via_relation(
            spans=spans,
            relations=relations,
            link_relation_label=self.link_relation_label,
            create_multi_spans=self.create_multi_spans,
        )

        result = document.copy(with_annotations=False).as_type(new_type=self.result_document_type)
        span_layer_name = document[self.relation_layer].target_name
        result_span_layer_name = self.result_field_mapping[span_layer_name]
        result_relation_layer_name = self.result_field_mapping[self.relation_layer]
        result[result_span_layer_name].extend(new_spans)
        result[result_relation_layer_name].extend(new_relations)

        return result


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


def get_common_pipeline_steps_with_merge_multi_spans(
    target_document_type: type[Document],
) -> dict:
    return dict(
        merge_spans=SpansWithRelationsMerger(
            relation_layer="relations",
            link_relation_label="parts_of_same",
            create_multi_spans=False,
            result_document_type=target_document_type,
            result_field_mapping={"spans": "labeled_spans", "relations": "binary_relations"},
        ),
        trim_adus=TextSpanTrimmer(layer="labeled_spans"),
        sort_symmetric_relation_arguments=RelationArgumentSorter(
            relation_layer="binary_relations",
            label_whitelist=["parts_of_same", "semantically_same"],
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
        "resolve_parts_of_same": BratDocumentWithMergedSpans,
    }

    # we need to add None to the list of dataset variants to support the default dataset variant
    BASE_BUILDER_KWARGS_DICT = {
        dataset_variant: {"url": URL, "split_paths": SPLIT_PATHS}
        for dataset_variant in ["default", "resolve_parts_of_same", None]
    }

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
                    **get_common_pipeline_steps(TextDocumentWithLabeledSpansAndBinaryRelations)
                ),
                TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions: Pipeline(
                    **get_common_pipeline_steps(
                        TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions
                    ),
                    add_partitions=regex_partitioner,
                ),
            }
        else:
            return {
                TextDocumentWithLabeledSpansAndBinaryRelations: Pipeline(
                    **get_common_pipeline_steps_with_merge_multi_spans(
                        TextDocumentWithLabeledSpansAndBinaryRelations
                    )
                ),
                TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions: Pipeline(
                    **get_common_pipeline_steps_with_merge_multi_spans(
                        TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions
                    ),
                    add_partitions=regex_partitioner,
                ),
                # TODO: add TextDocumentWithLabeledMultiSpansAndBinaryRelations
                # TODO: add TextDocumentWithLabeledMultiSpansBinaryRelationsAndLabeledPartitions
            }
