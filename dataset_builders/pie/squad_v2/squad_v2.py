import dataclasses
from typing import Any, Dict, Optional

import datasets
from pie_documents.annotations import ExtractiveAnswer, Question
from pie_documents.documents import ExtractiveQADocument

from pie_datasets import GeneratorBasedBuilder


@dataclasses.dataclass
class SquadV2Document(ExtractiveQADocument):
    """A PIE document with annotations for SQuAD v2.0."""

    title: Optional[str] = None


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

    DOCUMENT_CONVERTERS = {
        ExtractiveQADocument: {},  # no conversion required, just cast to the correct type
    }

    def _generate_document(self, example, **kwargs):
        return example_to_document(example)

    def _generate_example(self, document, **kwargs):
        return document_to_example(document)
