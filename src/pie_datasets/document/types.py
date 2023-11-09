import dataclasses
import logging
from typing import Any, Dict, Optional

from pytorch_ie.annotations import BinaryRelation, LabeledSpan
from pytorch_ie.core import Annotation, AnnotationList, Document, annotation_field
from pytorch_ie.documents import TextBasedDocument, TokenBasedDocument

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TokenDocumentWithLabeledSpans(TokenBasedDocument):
    labeled_spans: AnnotationList[LabeledSpan] = annotation_field(target="tokens")


@dataclasses.dataclass
class TokenDocumentWithLabeledSpansAndBinaryRelations(TokenDocumentWithLabeledSpans):
    binary_relations: AnnotationList[BinaryRelation] = annotation_field(target="labeled_spans")
