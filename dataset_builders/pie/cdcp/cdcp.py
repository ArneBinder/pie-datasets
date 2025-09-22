import dataclasses
import logging
from typing import Any, Dict, List, Optional

import datasets
from pie_core import Annotation, AnnotationLayer, annotation_field
from pie_documents.annotations import BinaryRelation, LabeledSpan
from pie_documents.document.processing.text_span_trimmer import trim_text_spans
from pie_documents.documents import (
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
class Attribute(Annotation):
    value: str
    annotation: Annotation


@dataclasses.dataclass
class CDCPDocument(TextBasedDocument):
    propositions: AnnotationLayer[LabeledSpan] = annotation_field(target="text")
    relations: AnnotationLayer[BinaryRelation] = annotation_field(target="propositions")
    urls: AnnotationLayer[Attribute] = annotation_field(target="propositions")


def example_to_document(
    example: Dict[str, Any],
    relation_label: datasets.ClassLabel,
    proposition_label: datasets.ClassLabel,
):
    document = CDCPDocument(id=example["id"], text=example["text"])
    for proposition_dict in dl2ld(example["propositions"]):
        proposition = LabeledSpan(
            start=proposition_dict["start"],
            end=proposition_dict["end"],
            label=proposition_label.int2str(proposition_dict["label"]),
        )
        document.propositions.append(proposition)
        if proposition_dict.get("url", "") != "":
            url = Attribute(annotation=proposition, value=proposition_dict["url"])
            document.urls.append(url)

    for relation_dict in dl2ld(example["relations"]):
        relation = BinaryRelation(
            head=document.propositions[relation_dict["head"]],
            tail=document.propositions[relation_dict["tail"]],
            label=relation_label.int2str(relation_dict["label"]),
        )
        document.relations.append(relation)

    return document


def document_to_example(
    document: CDCPDocument,
    relation_label: datasets.ClassLabel,
    proposition_label: datasets.ClassLabel,
) -> Dict[str, Any]:
    result = {"id": document.id, "text": document.text}
    proposition2dict = {}
    proposition2idx = {}
    for idx, proposition in enumerate(document.propositions):
        proposition2dict[proposition] = {
            "start": proposition.start,
            "end": proposition.end,
            "label": proposition_label.str2int(proposition.label),
            "url": "",
        }
        proposition2idx[proposition] = idx
    for url in document.urls:
        proposition2dict[url.annotation]["url"] = url.value

    result["propositions"] = ld2dl(
        proposition2dict.values(), keys=["start", "end", "label", "url"]
    )

    relations = [
        {
            "head": proposition2idx[relation.head],
            "tail": proposition2idx[relation.tail],
            "label": relation_label.str2int(relation.label),
        }
        for relation in document.relations
    ]
    result["relations"] = ld2dl(relations, keys=["head", "tail", "label"])

    return result


def convert_to_text_document_with_labeled_spans_and_binary_relations(
    document: CDCPDocument,
    verbose: bool = True,
) -> TextDocumentWithLabeledSpansAndBinaryRelations:
    doc_simplified = document.as_type(
        TextDocumentWithLabeledSpansAndBinaryRelations,
        field_mapping={"propositions": "labeled_spans", "relations": "binary_relations"},
    )
    result = trim_text_spans(
        doc_simplified,
        layer="labeled_spans",
        verbose=verbose,
    )
    return result


class CDCP(GeneratorBasedBuilder):
    DOCUMENT_TYPE = CDCPDocument

    DOCUMENT_CONVERTERS = {
        TextDocumentWithLabeledSpansAndBinaryRelations: convert_to_text_document_with_labeled_spans_and_binary_relations
    }

    BASE_DATASET_PATH = "DFKI-SLT/cdcp"
    BASE_DATASET_REVISION = "3cf79257900b3f97e4b8f9faae2484b1a534f484"

    BUILDER_CONFIGS = [datasets.BuilderConfig(name="default")]

    DEFAULT_CONFIG_NAME = "default"  # type: ignore

    def _generate_document_kwargs(self, dataset):
        return {
            "relation_label": dataset.features["relations"].feature["label"],
            "proposition_label": dataset.features["propositions"].feature["label"],
        }

    def _generate_document(self, example, relation_label, proposition_label):
        return example_to_document(
            example, relation_label=relation_label, proposition_label=proposition_label
        )
