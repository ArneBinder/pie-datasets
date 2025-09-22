import dataclasses
from typing import Any, Dict, List

import datasets
from pie_core import Annotation, AnnotationLayer, annotation_field
from pie_documents.documents import TextBasedDocument

from pie_datasets import GeneratorBasedBuilder


@dataclasses.dataclass(eq=True, frozen=True)
class AbstractiveSummary(Annotation):
    """A question about a context."""

    text: str

    def __str__(self) -> str:
        return self.text


@dataclasses.dataclass(eq=True, frozen=True)
class SectionName(Annotation):
    """A question about a context."""

    text: str

    def __str__(self) -> str:
        return self.text


@dataclasses.dataclass
class ScientificPapersDocument(TextBasedDocument):
    """A PIE document for scientific papers dataset."""

    abstract: AnnotationLayer[AbstractiveSummary] = annotation_field()
    section_names: AnnotationLayer[SectionName] = annotation_field()


def example_to_document(
    example: Dict[str, Any],
) -> ScientificPapersDocument:
    """Convert a Huggingface Scientific Papers example to a PIE document."""
    document = ScientificPapersDocument(
        text=example["article"],
    )
    document.abstract.append(AbstractiveSummary(text=example["abstract"]))
    document.section_names.extend(
        [SectionName(text=section_name) for section_name in example["section_names"].split("\n")]
    )

    return document


def document_to_example(doc: ScientificPapersDocument) -> Dict[str, Any]:
    """Convert a PIE document to a Huggingface Scientific Papers example."""
    example = {
        "article": doc.text,
        "abstract": doc.abstract[0].text,
        "section_names": "\n".join([section_name.text for section_name in doc.section_names]),
    }
    return example


class ScientificPapersConfig(datasets.BuilderConfig):
    """BuilderConfig for Scientific Papers."""

    def __init__(self, **kwargs):
        """BuilderConfig for Scientific Papers.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super().__init__(**kwargs)


class ScientificPapers(GeneratorBasedBuilder):
    DOCUMENT_TYPE = ScientificPapersDocument

    BASE_DATASET_PATH = "scientific_papers"
    BASE_DATASET_REVISION = "14c5296f2d707630f5835c9da59dcaddeea19b20"

    BUILDER_CONFIGS = [
        ScientificPapersConfig(
            name="arxiv",
            version=datasets.Version("1.1.1"),
            description="Scientific Papers dataset - ArXiv variant",
        ),
        ScientificPapersConfig(
            name="pubmed",
            version=datasets.Version("1.1.1"),
            description="Scientific Papers dataset - PubMed variant",
        ),
    ]

    DEFAULT_CONFIG_NAME = "arxiv"

    def _generate_document(self, example, **kwargs):
        return example_to_document(example)

    def _generate_example(self, document, **kwargs):
        return document_to_example(document)
