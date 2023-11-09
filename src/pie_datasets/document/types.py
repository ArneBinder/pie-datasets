import dataclasses
from typing import Optional

from pytorch_ie.annotations import BinaryRelation, LabeledMultiSpan, LabeledSpan
from pytorch_ie.core import Annotation, AnnotationList, annotation_field
from pytorch_ie.documents import TextBasedDocument, TokenBasedDocument


@dataclasses.dataclass(eq=True, frozen=True)
class Attribute(Annotation):
    annotation: Annotation
    label: str
    value: Optional[str] = None
    score: Optional[float] = dataclasses.field(default=None, compare=False)


@dataclasses.dataclass
class BratDocument(TextBasedDocument):
    spans: AnnotationList[LabeledMultiSpan] = annotation_field(target="text")
    relations: AnnotationList[BinaryRelation] = annotation_field(target="spans")
    span_attributes: AnnotationList[Attribute] = annotation_field(target="spans")
    relation_attributes: AnnotationList[Attribute] = annotation_field(target="relations")


@dataclasses.dataclass
class BratDocumentWithMergedSpans(TextBasedDocument):
    spans: AnnotationList[LabeledSpan] = annotation_field(target="text")
    relations: AnnotationList[BinaryRelation] = annotation_field(target="spans")
    span_attributes: AnnotationList[Attribute] = annotation_field(target="spans")
    relation_attributes: AnnotationList[Attribute] = annotation_field(target="relations")


@dataclasses.dataclass
class TokenDocumentWithLabeledSpans(TokenBasedDocument):
    labeled_spans: AnnotationList[LabeledSpan] = annotation_field(target="tokens")


@dataclasses.dataclass
class TokenDocumentWithLabeledSpansAndBinaryRelations(TokenDocumentWithLabeledSpans):
    binary_relations: AnnotationList[BinaryRelation] = annotation_field(target="labeled_spans")
