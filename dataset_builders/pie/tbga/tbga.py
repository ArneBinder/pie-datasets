import dataclasses
from typing import Any

from pytorch_ie import AnnotationLayer, annotation_field
from pytorch_ie.annotations import BinaryRelation, Span
from pytorch_ie.documents import TextBasedDocument


@dataclasses.dataclass(frozen=True)
class SpanWithIdAndName(Span):
    id: str
    name: str

    def resolve(self) -> Any:
        return self.id, self.name, super().resolve()


@dataclasses.dataclass
class BioRelDocument(TextBasedDocument):
    entities: AnnotationLayer[SpanWithIdAndName] = annotation_field(target="text")
    relations: AnnotationLayer[BinaryRelation] = annotation_field(target="entities")
