import dataclasses
import logging
from typing import Any, Callable, Dict, List, Optional

import datasets
import pytorch_ie.data.builder
from pytorch_ie.annotations import BinaryRelation, LabeledSpan
from pytorch_ie.core import Annotation, AnnotationList, Document, annotation_field
from pytorch_ie.documents import TextDocumentWithLabeledSpansAndBinaryRelations

from pie_datasets.document.processing.text_span_trimmer import trim_text_spans

log = logging.getLogger(__name__)


def dl2ld(dict_of_lists):
    return [dict(zip(dict_of_lists, t)) for t in zip(*dict_of_lists.values())]


def ld2dl(list_of_dicts, keys: Optional[List[str]] = None, as_list: bool = False):
    if keys is None:
        keys = list_of_dicts[0].keys()
    if as_list:
        return [[d[k] for d in list_of_dicts] for k in keys]
    else:
        return {k: [d[k] for d in list_of_dicts] for k in keys}


@dataclasses.dataclass(frozen=True)
class Attribute(Annotation):
    value: str
    annotation: Annotation


@dataclasses.dataclass
class CDCPDocument(Document):
    text: str
    id: Optional[str] = None
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)
    propositions: AnnotationList[LabeledSpan] = annotation_field(target="text")
    relations: AnnotationList[BinaryRelation] = annotation_field(target="propositions")
    urls: AnnotationList[Attribute] = annotation_field(target="propositions")


def example_to_document(
    example: Dict[str, Any],
    relation_int2str: Callable[[int], str],
    proposition_int2str: Callable[[int], str],
):
    document = CDCPDocument(id=example["id"], text=example["text"])
    for proposition_dict in dl2ld(example["propositions"]):
        proposition = LabeledSpan(
            start=proposition_dict["start"],
            end=proposition_dict["end"],
            label=proposition_int2str(proposition_dict["label"]),
        )
        document.propositions.append(proposition)
        if proposition_dict.get("url", "") != "":
            url = Attribute(annotation=proposition, value=proposition_dict["url"])
            document.urls.append(url)

    for relation_dict in dl2ld(example["relations"]):
        relation = BinaryRelation(
            head=document.propositions[relation_dict["head"]],
            tail=document.propositions[relation_dict["tail"]],
            label=relation_int2str(relation_dict["label"]),
        )
        document.relations.append(relation)

    return document


def document_to_example(
    document: CDCPDocument,
    relation_str2int: Callable[[str], int],
    proposition_str2int: Callable[[str], int],
) -> Dict[str, Any]:
    result = {"id": document.id, "text": document.text}
    proposition2dict = {}
    proposition2idx = {}
    for idx, proposition in enumerate(document.propositions):
        proposition2dict[proposition] = {
            "start": proposition.start,
            "end": proposition.end,
            "label": proposition_str2int(proposition.label),
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
            "label": relation_str2int(relation.label),
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


class CDCPConfig(datasets.BuilderConfig):
    """BuilderConfig for CDCP."""

    def __init__(self, **kwargs):
        """BuilderConfig for CDCP.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super().__init__(**kwargs)


class CDCP(pytorch_ie.data.builder.GeneratorBasedBuilder):
    DOCUMENT_TYPE = CDCPDocument

    DOCUMENT_CONVERTERS = {
        TextDocumentWithLabeledSpansAndBinaryRelations: convert_to_text_document_with_labeled_spans_and_binary_relations
    }

    BASE_DATASET_PATH = "DFKI-SLT/cdcp"

    BUILDER_CONFIGS = [datasets.BuilderConfig(name="default")]

    DEFAULT_CONFIG_NAME = "default"  # type: ignore

    def _generate_document_kwargs(self, dataset):
        return {
            "relation_int2str": dataset.features["relations"].feature["label"].int2str,
            "proposition_int2str": dataset.features["propositions"].feature["label"].int2str,
        }

    def _generate_document(self, example, relation_int2str, proposition_int2str):
        return example_to_document(
            example, relation_int2str=relation_int2str, proposition_int2str=proposition_int2str
        )
