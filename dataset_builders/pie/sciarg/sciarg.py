from __future__ import annotations

import logging
from typing import TypeVar

from pytorch_ie.annotations import BinaryRelation, LabeledSpan
from pytorch_ie.core import AnnotationList, Document
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
    TextSpanTrimmer,
)

URL = "http://data.dws.informatik.uni-mannheim.de/sci-arg/compiled_corpus.zip"
SUBDIRECTORY_MAPPING = {"compiled_corpus": "train"}


logger = logging.getLogger(__name__)


D = TypeVar("D", bound=Document)


class RelationArgumentSorter:
    def __init__(self, relation_layer: str, labels: list[str] | None = None):
        self.relation_layer = relation_layer
        self.labels = labels

    def __call__(self, doc: D) -> D:
        rel_layer: AnnotationList[BinaryRelation] = doc[self.relation_layer]
        args2relations: dict[tuple[LabeledSpan, LabeledSpan], BinaryRelation] = {
            (rel.head, rel.tail): rel for rel in rel_layer
        }
        # TODO: assert that no other layers depend on the relation layer
        rel_layer.clear()
        for args, rel in args2relations.items():
            if self.labels is None or rel.label in self.labels:
                if not isinstance(rel, BinaryRelation):
                    raise TypeError(
                        f"can only sort arguments of BinaryRelations, but relations is: {rel}"
                    )
                new_head, new_tail = tuple(sorted(args, key=lambda arg: (arg.start, arg.end)))
                if args != (new_head, new_tail):
                    if (new_head, new_tail) in args2relations:
                        prev_rel = args2relations[(new_head, new_tail)]
                        if prev_rel.label != rel.label:
                            raise ValueError(
                                f"there is already a relation with sorted args {(new_head, new_tail)} "
                                f"but with a different label: {prev_rel.label} != {rel.label}"
                            )
                        else:
                            logger.warning(
                                f"do not add the new relation with sorted arguments, because it is already there: "
                                f"{prev_rel}"
                            )
                    else:
                        new_rel = BinaryRelation(
                            head=new_head,
                            tail=new_tail,
                            label=rel.label,
                            score=rel.score,
                        )
                        rel_layer.append(new_rel)
                else:
                    # if the relation args are already sorted, just add the relation
                    rel_layer.append(rel)
            else:
                # just add the relations with a different label
                rel_layer.append(rel)

        # sanity check (TODO: should be in a test instead)
        for rel in rel_layer:
            # check that the args are sorted
            if (self.labels is None or rel.label in self.labels) and (
                rel.head.start,
                rel.head.end,
            ) > (rel.tail.start, rel.tail.end):
                raise ValueError(f"relation args are not sorted: {rel}")

        return doc


def get_common_pipeline_steps(target_document_type: type[Document]) -> dict:
    return dict(
        cast=Caster(
            document_type=target_document_type,
            field_mapping={"spans": "labeled_spans", "relations": "binary_relations"},
        ),
        trim_adus=TextSpanTrimmer(layer="labeled_spans"),
        sort_symmetric_relation_arguments=RelationArgumentSorter(
            relation_layer="binary_relations", labels=["parts_of_same", "semantically_same"]
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
            raise NotImplementedError(
                'SciArg does not support document converters for the "default" dataset variant. '
                'Consider using name="merge_fragmented_spans" instead.'
            )
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
