import dataclasses
from typing import Any, Dict, Optional

import datasets
from pytorch_ie.annotations import Span
from pytorch_ie.core import Annotation, AnnotationList, annotation_field
from pytorch_ie.documents import TextBasedDocument

from pie_datasets import GeneratorBasedBuilder


@dataclasses.dataclass(eq=True, frozen=True)
class Question(Annotation):
    """A question about a context."""

    text: str

    def __str__(self) -> str:
        return self.text


@dataclasses.dataclass(eq=True, frozen=True)
class ExtractiveAnswer(Span):
    """An answer to a question."""

    # this annotation has two target fields
    TARGET_NAMES = ("context", "questions")

    question: Question

    def __str__(self) -> str:
        if not self.is_attached:
            return ""
        context = self.named_targets["context"]
        return str(context[self.start : self.end])


@dataclasses.dataclass
class SquadV2Document(TextBasedDocument):
    """A PIE document with annotations for SQuAD v2.0."""

    title: Optional[str] = None
    questions: AnnotationList[Question] = annotation_field()
    answers: AnnotationList[ExtractiveAnswer] = annotation_field(
        # The answers annotation layer depends on two other data fields / layers:
        # The "text" data field (this is derived from TextBasedDocument) and the "questions" annotation layer.
        # Any annotation layer with ExtractiveAnswer annotations expects the targets "context" and "questions".
        # We provide the respective mapping as "named_targets".
        named_targets={"context": "text", "questions": "questions"}
    )


def example_to_document(
    example: Dict[str, Any],
) -> SquadV2Document:
    """Convert a Huggingface SQuAD v2.0 example to a PIE document."""
    document = SquadV2Document(
        id=example["id"],
        title=example["title"],
        text=example["context"],
    )
    question = Question(example["question"])
    document.questions.append(question)
    for answer_text, answer_start in zip(
        example["answers"]["text"], example["answers"]["answer_start"]
    ):
        answer = ExtractiveAnswer(
            question=question, start=answer_start, end=answer_start + len(answer_text)
        )
        document.answers.append(answer)
    return document


def document_to_example(doc: SquadV2Document) -> Dict[str, Any]:
    """Convert a PIE document to a Huggingface SQuAD v2.0 example."""
    example = {
        "id": doc.id,
        "title": doc.title,
        "context": doc.text,
        "question": doc.questions[0].text,
        "answers": {
            "text": [str(a) for a in doc.answers],
            "answer_start": [a.start for a in doc.answers],
        },
    }
    return example


class SquadV2Config(datasets.BuilderConfig):
    """BuilderConfig for SQuAD v2.0."""

    def __init__(self, **kwargs):
        """BuilderConfig for SQuAD v2.0.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super().__init__(**kwargs)


class SquadV2(GeneratorBasedBuilder):
    DOCUMENT_TYPE = SquadV2Document

    BASE_DATASET_PATH = "squad_v2"
    BASE_DATASET_REVISION = "e4d7191788b08fde3cbd09bd8fe1fcd827ee1715"

    BUILDER_CONFIGS = [
        SquadV2Config(
            name="squad_v2",
            version=datasets.Version("2.0.0"),
            description="SQuAD plain text version 2",
        ),
    ]

    DEFAULT_CONFIG_NAME = "squad_v2"

    def _generate_document(self, example):
        return example_to_document(example)
