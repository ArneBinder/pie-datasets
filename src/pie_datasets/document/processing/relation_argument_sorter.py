from __future__ import annotations

import logging
from typing import TypeVar

from pytorch_ie.annotations import BinaryRelation, LabeledSpan
from pytorch_ie.core import Annotation, AnnotationList, Document

logger = logging.getLogger(__name__)


D = TypeVar("D", bound=Document)


def get_relation_args(relation: Annotation) -> tuple[Annotation, ...]:
    if isinstance(relation, BinaryRelation):
        return relation.head, relation.tail
    else:
        raise TypeError(
            f"relation {relation} has unknown type [{type(relation)}], cannot get arguments from it"
        )


def construct_relation_with_new_args(
    relation: Annotation, new_args: tuple[Annotation, ...]
) -> BinaryRelation:
    if isinstance(relation, BinaryRelation):
        return BinaryRelation(
            head=new_args[0],
            tail=new_args[1],
            label=relation.label,
            score=relation.score,
        )
    else:
        raise TypeError(
            f"original relation {relation} has unknown type [{type(relation)}], "
            f"cannot reconstruct it with new arguments"
        )


def has_dependent_layers(document: D, layer: str) -> bool:
    return layer not in document._annotation_graph["_artificial_root"]


class RelationArgumentSorter:
    def __init__(
        self, relation_layer: str, label_blacklist: list[str] | None = None, inplace: bool = True
    ):
        self.relation_layer = relation_layer
        self.label_blacklist = label_blacklist
        self.inplace = inplace

    def __call__(self, doc: D) -> D:
        if not self.inplace:
            doc = doc.copy()

        rel_layer: AnnotationList[BinaryRelation] = doc[self.relation_layer]
        args2relations: dict[tuple[LabeledSpan, ...], BinaryRelation] = {
            get_relation_args(rel): rel for rel in rel_layer
        }

        # assert that no other layers depend on the relation layer
        if has_dependent_layers(document=doc, layer=self.relation_layer):
            raise ValueError(
                f"the relation layer {self.relation_layer} has dependent layers, "
                f"cannot sort the arguments of the relations"
            )

        rel_layer.clear()
        for args, rel in args2relations.items():
            if self.label_blacklist is not None and rel.label in self.label_blacklist:
                # just add the relations whose label is not in the label blacklist (if a blacklist is present)
                rel_layer.append(rel)
            else:
                args_sorted = tuple(sorted(args, key=lambda arg: (arg.start, arg.end)))
                if args == args_sorted:
                    # if the relation args are already sorted, just add the relation
                    rel_layer.append(rel)
                else:
                    if args_sorted not in args2relations:
                        new_rel = construct_relation_with_new_args(rel, args_sorted)
                        rel_layer.append(new_rel)
                    else:
                        prev_rel = args2relations[args_sorted]
                        if prev_rel.label != rel.label:
                            raise ValueError(
                                f"there is already a relation with sorted args {args_sorted} "
                                f"but with a different label: {prev_rel.label} != {rel.label}"
                            )
                        else:
                            logger.warning(
                                f"do not add the new relation with sorted arguments, because it is already there: "
                                f"{prev_rel}"
                            )

        return doc
