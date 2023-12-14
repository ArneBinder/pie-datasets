import dataclasses
from typing import Any, Dict

import datasets
from pytorch_ie.core import annotation_field
from pytorch_ie.documents import TextBasedDocument

from pie_datasets import GeneratorBasedBuilder


@dataclasses.dataclass
class ScientificPapersDocument(TextBasedDocument):
    """A PIE document for scientific papers dataset."""

    abstract: str = annotation_field()
    section_names: str = annotation_field()


def example_to_document(
    example: Dict[str, Any],
) -> ScientificPapersDocument:
    """Convert a Huggingface Scientific Papers example to a PIE document."""
    document = ScientificPapersDocument(
        text=example["article"],
    )
    document.abstract = example["abstract"]
    document.section_names = example["section_names"].split("\n")

    return document


def document_to_example(doc: ScientificPapersDocument) -> Dict[str, Any]:
    """Convert a PIE document to a Huggingface Scientific Papers example."""
    example = {
        "article": doc.text,
        "abstract": doc.abstract,
        "section_names": "\n".join(doc.section_names),
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

    def _generate_document_kwargs(self, example, **kwargs):
        return {}

    def _generate_document(self, example, **kwargs):
        return example_to_document(example)

    def _generate_example(self, document, **kwargs):
        return document_to_example(document)
