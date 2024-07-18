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
class TbgaDocument(TextBasedDocument):
    entities: AnnotationLayer[SpanWithIdAndName] = annotation_field(target="text")
    relations: AnnotationLayer[BinaryRelation] = annotation_field(target="entities")


def example_to_document(example) -> TbgaDocument:
    document = TbgaDocument(text=example["text"])
    head = SpanWithIdAndName(
        id=example["h"]["id"],
        name=example["h"]["name"],
        start=example["h"]["pos"][0],
        end=example["h"]["pos"][1],
    )
    tail = SpanWithIdAndName(
        id=example["t"]["id"],
        name=example["t"]["name"],
        start=example["t"]["pos"][0],
        end=example["t"]["pos"][1],
    )
    document.entities.extend([head, tail])

    relation = BinaryRelation(head=head, tail=tail, label=example["relation"])
    document.relations.append(relation)
    return document