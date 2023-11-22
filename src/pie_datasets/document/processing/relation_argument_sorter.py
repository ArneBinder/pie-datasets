from __future__ import annotations

import logging
from typing import TypeVar

from pytorch_ie.annotations import BinaryRelation, LabeledSpan
from pytorch_ie.core import AnnotationList, Document

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
