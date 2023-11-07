import dataclasses
import logging
from typing import Any, Dict, Optional

from pytorch_ie.annotations import (
    BinaryRelation,
    LabeledMultiSpan,
    LabeledSpan,
    Span,
    _post_init_single_label,
)
from pytorch_ie.core import Annotation, AnnotationList, Document, annotation_field
from pytorch_ie.documents import TextBasedDocument, TokenBasedDocument

logger = logging.getLogger(__name__)


# ========================= Annotation Types ========================= #


@dataclasses.dataclass(eq=True, frozen=True)
class Attribute(Annotation):
    target_annotation: Annotation
    label: str
    value: Optional[str] = None
    score: float = 1.0

    def __post_init__(self) -> None:
        _post_init_single_label(self)


# ========================= Document Types ========================= #


@dataclasses.dataclass
class TokenDocumentWithLabeledSpans(TokenBasedDocument):
    labeled_spans: AnnotationList[LabeledSpan] = annotation_field(target="tokens")


@dataclasses.dataclass
class TokenDocumentWithLabeledSpansAndBinaryRelations(TokenDocumentWithLabeledSpans):
    binary_relations: AnnotationList[BinaryRelation] = annotation_field(target="labeled_spans")
