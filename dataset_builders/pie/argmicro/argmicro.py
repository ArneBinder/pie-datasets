import copy
import dataclasses
import logging
from collections import defaultdict
from itertools import combinations
from typing import Any, Dict, List, Optional, Set, Tuple

import datasets
from pytorch_ie.annotations import BinaryRelation, Label, LabeledSpan, Span
from pytorch_ie.core import Annotation, AnnotationList, annotation_field
from pytorch_ie.documents import (
    TextBasedDocument,
    TextDocumentWithLabeledSpansAndBinaryRelations,
)

from pie_datasets import GeneratorBasedBuilder

log = logging.getLogger(__name__)


def dl2ld(dict_of_lists):
    return [dict(zip(dict_of_lists, t)) for t in zip(*dict_of_lists.values())]


def ld2dl(list_of_dicts, keys: Optional[List[str]] = None):
    return {k: [d[k] for d in list_of_dicts] for k in keys}


@dataclasses.dataclass(frozen=True)
class LabeledAnnotationCollection(Annotation):
    annotations: Tuple[Annotation, ...]
    label: str


@dataclasses.dataclass(frozen=True)
class MultiRelation(Annotation):
    heads: Tuple[Annotation, ...]  # sources == heads
    tails: Tuple[Annotation, ...]  # targets == tails
    label: str


@dataclasses.dataclass
class ArgMicroDocument(TextBasedDocument):
    topic_id: Optional[str] = None
    stance: AnnotationList[Label] = annotation_field()
    edus: AnnotationList[Span] = annotation_field(target="text")
    adus: AnnotationList[LabeledAnnotationCollection] = annotation_field(target="edus")
    relations: AnnotationList[MultiRelation] = annotation_field(target="adus")


def example_to_document(
    example: Dict[str, Any],
    adu_type_label: datasets.ClassLabel,
    edge_type_label: datasets.ClassLabel,
    stance_label: datasets.ClassLabel,
) -> ArgMicroDocument:
    stance = stance_label.int2str(example["stance"])
    document = ArgMicroDocument(
        id=example["id"],
        text=example["text"],
        topic_id=example["topic_id"] if example["topic_id"] != "UNDEFINED" else None,
    )
    if stance != "UNDEFINED":
        document.stance.append(Label(label=stance))

    # build EDUs
    edus_dict = {
        edu["id"]: Span(start=edu["start"], end=edu["end"]) for edu in dl2ld(example["edus"])
    }
    # build ADUs
    adu_id2edus = defaultdict(list)
    edges_multi_source = defaultdict(dict)
    for edge in dl2ld(example["edges"]):
        edge_type = edge_type_label.int2str(edge["type"])
        if edge_type == "seg":
            adu_id2edus[edge["trg"]].append(edus_dict[edge["src"]])
        elif edge_type == "add":
            if "src" not in edges_multi_source[edge["trg"]]:
                edges_multi_source[edge["trg"]]["src"] = []
            edges_multi_source[edge["trg"]]["src"].append(edge["src"])
        else:
            edges_multi_source[edge["id"]]["type"] = edge_type
            edges_multi_source[edge["id"]]["trg"] = edge["trg"]
            if "src" not in edges_multi_source[edge["id"]]:
                edges_multi_source[edge["id"]]["src"] = []
            edges_multi_source[edge["id"]]["src"].append(edge["src"])
    adus_dict = {}
    for adu in dl2ld(example["adus"]):
        adu_type = adu_type_label.int2str(adu["type"])
        adu_edus = adu_id2edus[adu["id"]]
        adus_dict[adu["id"]] = LabeledAnnotationCollection(
            annotations=tuple(adu_edus), label=adu_type
        )
    # build relations
    rels_dict = {}
    for edge_id, edge in edges_multi_source.items():
        edge_target = edge["trg"]
        if edge_target in edges_multi_source:
            targets = edges_multi_source[edge_target]["src"]
        else:
            targets = [edge_target]
        if any(target in edges_multi_source for target in targets):
            raise Exception("Multi-hop relations are not supported")
        rel = MultiRelation(
            heads=tuple(adus_dict[source] for source in edge["src"]),
            tails=tuple(adus_dict[target] for target in targets),
            label=edge["type"],
        )
        rels_dict[edge_id] = rel

    document.edus.extend(edus_dict.values())
    document.adus.extend(adus_dict.values())
    document.relations.extend(rels_dict.values())
    document.metadata["edu_ids"] = list(edus_dict.keys())
    document.metadata["adu_ids"] = list(adus_dict.keys())
    document.metadata["rel_ids"] = list(rels_dict.keys())

    document.metadata["rel_seg_ids"] = {
        edge["src"]: edge["id"]
        for edge in dl2ld(example["edges"])
        if edge_type_label.int2str(edge["type"]) == "seg"
    }
    document.metadata["rel_add_ids"] = {
        edge["src"]: edge["id"]
        for edge in dl2ld(example["edges"])
        if edge_type_label.int2str(edge["type"]) == "add"
    }
    return document


def document_to_example(
    document: ArgMicroDocument,
    adu_type_label: datasets.ClassLabel,
    edge_type_label: datasets.ClassLabel,
    stance_label: datasets.ClassLabel,
) -> Dict[str, Any]:
    stance = document.stance[0].label if len(document.stance) else "UNDEFINED"
    result = {
        "id": document.id,
        "text": document.text,
        "topic_id": document.topic_id or "UNDEFINED",
        "stance": stance_label.str2int(stance),
    }

    # construct EDUs
    edus = {
        edu: {"id": edu_id, "start": edu.start, "end": edu.end}
        for edu_id, edu in zip(document.metadata["edu_ids"], document.edus)
    }
    result["edus"] = ld2dl(
        sorted(edus.values(), key=lambda x: x["id"]), keys=["id", "start", "end"]
    )

    # construct ADUs
    adus = {
        adu: {"id": adu_id, "type": adu_type_label.str2int(adu.label)}
        for adu_id, adu in zip(document.metadata["adu_ids"], document.adus)
    }
    result["adus"] = ld2dl(sorted(adus.values(), key=lambda x: x["id"]), keys=["id", "type"])

    # construct edges
    rels_dict: Dict[str, MultiRelation] = {
        rel_id: rel for rel_id, rel in zip(document.metadata["rel_ids"], document.relations)
    }
    heads2rel_id = {
        rel.heads: red_id for red_id, rel in zip(document.metadata["rel_ids"], document.relations)
    }
    edges = []
    for rel_id, rel in rels_dict.items():
        # if it is an undercut attack, we need to change the target to the relation that connects the target
        if rel.label == "und":
            target_id = heads2rel_id[rel.tails]
        else:
            if len(rel.tails) > 1:
                raise Exception("Multi-target relations are not supported")
            target_id = adus[rel.tails[0]]["id"]
        source_id = adus[rel.heads[0]]["id"]
        edge = {
            "id": rel_id,
            "src": source_id,
            "trg": target_id,
            "type": edge_type_label.str2int(rel.label),
        }
        edges.append(edge)
        # if it is an additional support, we need to change the source to the relation that connects the source
        for head in rel.heads[1:]:
            source_id = adus[head]["id"]
            edge_id = document.metadata["rel_add_ids"][source_id]
            edge = {
                "id": edge_id,
                "src": source_id,
                "trg": rel_id,
                "type": edge_type_label.str2int("add"),
            }
            edges.append(edge)

    for adu_id, adu in zip(document.metadata["adu_ids"], document.adus):
        for edu in adu.annotations:
            source_id = edus[edu]["id"]
            target_id = adus[adu]["id"]
            edge_id = document.metadata["rel_seg_ids"][source_id]
            edge = {
                "id": edge_id,
                "src": source_id,
                "trg": target_id,
                "type": edge_type_label.str2int("seg"),
            }
            edges.append(edge)

    result["edges"] = ld2dl(
        sorted(edges, key=lambda x: x["id"]), keys=["id", "src", "trg", "type"]
    )
    return result


def convert_to_text_document_with_labeled_spans_and_binary_relations(
    doc: ArgMicroDocument,
) -> TextDocumentWithLabeledSpansAndBinaryRelations:
    # convert adus to entities
    entities = []
    adu2entity: Dict[LabeledAnnotationCollection, Span] = {}
    for adu in doc.adus:
        edus: Set[Span] = set(adu.annotations)
        start = min(edu.start for edu in edus)
        end = max(edu.end for edu in edus)
        # assert there are no edus overlapping with the adu, but not part of it
        for edu in doc.edus:
            if (start <= edu.start < end or start < edu.end <= end) and edu not in edus:
                raise Exception(f"edu {edu} is overlapping with adu {adu}, but is not part of it")
        entity = LabeledSpan(start=start, end=end, label=adu.label)
        entities.append(entity)
        adu2entity[adu] = entity
    relations = []
    for relation in doc.relations:
        # add all possible combinations of heads and tails
        for head in relation.heads:
            for tail in relation.tails:
                rel = BinaryRelation(
                    label=relation.label, head=adu2entity[head], tail=adu2entity[tail]
                )
                relations.append(rel)
        # also add the relations between the heads themselves
        for head1, head2 in combinations(relation.heads, 2):
            rel = BinaryRelation(label="joint", head=adu2entity[head1], tail=adu2entity[head2])
            relations.append(rel)
            # also add the reverse relation
            rel = BinaryRelation(label="joint", head=adu2entity[head2], tail=adu2entity[head1])
            relations.append(rel)

    metadata = copy.deepcopy(doc.metadata)
    if len(doc.stance) > 0:
        metadata["stance"] = doc.stance[0].label
    metadata["topic"] = doc.topic_id
    result = TextDocumentWithLabeledSpansAndBinaryRelations(
        text=doc.text, id=doc.id, metadata=doc.metadata
    )
    result.labeled_spans.extend(entities)
    result.binary_relations.extend(relations)

    return result


class ArgMicro(GeneratorBasedBuilder):
    DOCUMENT_TYPE = ArgMicroDocument

    DOCUMENT_CONVERTERS = {
        TextDocumentWithLabeledSpansAndBinaryRelations: convert_to_text_document_with_labeled_spans_and_binary_relations
    }

    BASE_DATASET_PATH = "DFKI-SLT/argmicro"
    BASE_DATASET_REVISION = "22958d585f5c0c646c81ac62947bdf6cf9ab3cc5"

    BUILDER_CONFIGS = [datasets.BuilderConfig(name="en"), datasets.BuilderConfig(name="de")]

    def _generate_document_kwargs(self, dataset):
        return {
            "adu_type_label": dataset.features["adus"].feature["type"],
            "edge_type_label": dataset.features["edges"].feature["type"],
            "stance_label": dataset.features["stance"],
        }

    def _generate_document(self, example, **kwargs):
        return example_to_document(example, **kwargs)
